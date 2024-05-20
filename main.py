# -*- coding: utf-8 -*-
import torch
from config import load_args
from typing import Any, Dict, List, Tuple
from models import load_backbone
from os import path
from datasets import Features, load_dataset
from analytic import ACILLearner, DSALLearner, GKEALLearner
from torch.utils.data import Dataset, DataLoader, TensorDataset
from utils import set_determinism, validate
from tqdm import tqdm
from torchvision.transforms import transforms

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def make_dataloader(
    dataset: Dataset, shuffle: bool = False, batch_size: int = 256, num_workers: int = 8
) -> DataLoader:
    config = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": DEVICE.type == "cuda",
        "pin_memory_device": str(DEVICE) if DEVICE.type == "cuda" else "",
    }
    try:
        from prefetch_generator import BackgroundGenerator

        class DataLoaderX(DataLoader):
            def __iter__(self):
                return BackgroundGenerator(super().__iter__())

        return DataLoaderX(dataset, **config)
    except ImportError:
        return DataLoader(dataset, **config)

def check_cache_features(root: str) -> bool:
    files_list = ["X_train.pt", "y_train.pt", "X_test.pt", "y_test.pt"]
    for file in files_list:
        if not path.isfile(path.join(root, file)):
            return False
    return True

@torch.no_grad()
def cache_features(
    backbone: torch.nn.Module, dataloader: DataLoader[Tuple[torch.Tensor, torch.Tensor]]
) -> Tuple[torch.Tensor, torch.Tensor]:
    backbone.eval()
    X_all: List[torch.Tensor] = []
    y_all: List[torch.Tensor] = []
    for X, y in tqdm(dataloader, "Caching"):
        X: torch.Tensor = backbone(X.to(DEVICE))
        y: torch.Tensor = y.to(torch.int16, non_blocking=True)
        X_all.append(X.to("cpu", non_blocking=True))
        y_all.append(y)
    return torch.cat(X_all), torch.cat(y_all)

def main(args: Dict[str, Any]):
    print(f"Received augment_count: {args['augment_count']}")  
    backbone_name = args["backbone"]

    if args["seed"] is not None:
        set_determinism(args["seed"])

    if ("backbone_path" in args) and path.isfile(args["backbone_path"]):
        preload_backbone = True
        backbone, _, feature_size = torch.load(
            args["backbone_path"], map_location=DEVICE
        )
    else:
        preload_backbone = False
        load_pretrain = args["base_ratio"] == 0 or "ImageNet" not in args["dataset"]
        backbone, _, feature_size = load_backbone(backbone_name, pretrain=load_pretrain)
        if load_pretrain:
            assert args["dataset"] != "ImageNet", "Data may leak!!!"
    backbone = backbone.to(DEVICE, non_blocking=True)

    dataset_args = {
        "name": args["dataset"],
        "root": args["data_root"],
        "base_ratio": args["base_ratio"],
        "num_phases": args["phases"],
        "shuffle_seed": args["dataset_seed"] if "dataset_seed" in args else None,
        "num_shots": args.get("num_shots", 5),
        "augment_count": args.get("augment_count", 200)
    }
    dataset_train = load_dataset(train=True, augment=True, **dataset_args)
    dataset_test = load_dataset(train=False, augment=False, **dataset_args)

    if args["method"] == "ACIL" or args["method"] == "G-ACIL":
        learner = ACILLearner(args, backbone, feature_size, DEVICE)
    elif args["method"] == "DS-AL":
        learner = DSALLearner(args, backbone, feature_size, DEVICE)
    elif args["method"] == "GKEAL":
        learner = GKEALLearner(args, backbone, feature_size, DEVICE)
    else:
        raise ValueError(f"Unknown method: {args['method']}")

    if args["base_ratio"] > 0 and not preload_backbone:
        train_subset = dataset_train.subset_at_phase(0)
        test_subset = dataset_test.subset_at_phase(0)
        train_loader = make_dataloader(
            train_subset, True, args["batch_size"], args["num_workers"]
        )
        test_loader = make_dataloader(
            test_subset, False, args["batch_size"], args["num_workers"]
        )
        learner.base_training(
            train_loader,
            test_loader,
            dataset_train.base_size,
        )

    if args["cache_features"]:
        if "cache_path" not in args:
            args["cache_path"] = args["saving_root"]
        if not check_cache_features(args["cache_path"]):
            backbone = learner.backbone.eval()
            dataset_train = load_dataset(
                args["dataset"], args["data_root"], True, 1, 0, augment=False
            )
            dataset_test = load_dataset(
                args["dataset"], args["data_root"], False, 1, 0, augment=False
            )
            train_loader = make_dataloader(
                dataset_train.subset_at_phase(0),
                False,
                args["batch_size"],
                args["num_workers"],
            )
            test_loader = make_dataloader(
                dataset_test.subset_at_phase(0),
                False,
                args["batch_size"],
                args["num_workers"],
            )

            X_train, y_train = cache_features(backbone, train_loader)
            X_test, y_test = cache_features(backbone, test_loader)
            torch.save(X_train, path.join(args["cache_path"], "X_train.pt"))
            torch.save(y_train, path.join(args["cache_path"], "y_train.pt"))
            torch.save(X_test, path.join(args["cache_path"], "X_test.pt"))
            torch.save(y_test, path.join(args["cache_path"], "y_test.pt"))
        dataset_train = Features(
            args["cache_path"],
            train=True,
            base_ratio=args["base_ratio"],
            num_phases=args["phases"],
            augment=False,
        )
        dataset_test = Features(
            args["cache_path"],
            train=False,
            base_ratio=args["base_ratio"],
            num_phases=args["phases"],
            augment=False,
        )
        learner.backbone = torch.nn.Identity()
        learner.model.backbone = torch.nn.Identity()
    else:
        dataset_train = load_dataset(train=True, augment=False, **dataset_args)
        dataset_test = load_dataset(train=False, augment=False, **dataset_args)

    sum_acc = 0
    log_file_path = path.join(args["saving_root"], "IL.csv")
    log_file = open(log_file_path, "w", buffering=1)
    print(
        "phase", "acc@avg", "acc@1", "acc@5", "f1-micro", "loss", file=log_file, sep=","
    )

    for phase in range(0, args["phases"] + 1):
        train_subset = dataset_train.subset_at_phase(phase)
        test_subset = dataset_test.subset_until_phase(phase)
        train_loader = make_dataloader(
            train_subset, True, args["IL_batch_size"], args["num_workers"]
        )
        test_loader = make_dataloader(
            test_subset, False, args["IL_batch_size"], args["num_workers"]
        )

        if phase == 0:
            learner.learn(train_loader, dataset_train.base_size, "Re-align")
        else:
            if args["use_afc"]:
                few_shot_data = dataset_train.few_shot_sampler(phase, random_seed=phase)
                # few_shot_data = train_subset
                few_shot_loader = make_dataloader(few_shot_data, True, args["IL_batch_size"], args["num_workers"])

                X_all_augmented = []
                y_all_augmented = []
                for X, y in few_shot_loader:
                    X_augmented = dataset_train.augment_data(X)
                    y_augmented = y.repeat_interleave(dataset_train.augment_count)

                    X_all_augmented.append(X_augmented)
                    y_all_augmented.append(y_augmented)

                X_all_augmented = torch.cat(X_all_augmented, dim=0)
                y_all_augmented = torch.cat(y_all_augmented, dim=0)

                print(f"X_all_augmented shape: {X_all_augmented.shape}, y_all_augmented shape: {y_all_augmented.shape}")
          
                augmented_dataset = TensorDataset(X_all_augmented, y_all_augmented)
                augmented_loader = make_dataloader(augmented_dataset, True, args["IL_batch_size"], args["num_workers"])
                print(len(augmented_loader.dataset))

                base_features = [dataset_train[i] for i in range(len(train_loader.dataset))]
                X_base, y_base = zip(*base_features)
                X_base = torch.stack(X_base)
                y_base = torch.tensor(y_base)

                print(f"X_base shape: {X_base.shape}, y_base shape: {y_base.shape}")

                X_combined = torch.cat((X_all_augmented, X_base), dim=0)
                y_combined = torch.cat((y_all_augmented, y_base), dim=0)

                print(f"X_combined shape: {X_combined.shape}, y_combined shape: {y_combined.shape}")

                combined_dataset = TensorDataset(X_combined, y_combined)
                combined_loader = make_dataloader(combined_dataset, True, args["IL_batch_size"], args["num_workers"])
                print(len(combined_loader.dataset))
                learner.learn(combined_loader, len(combined_loader.dataset))
            else:
                learner.learn(train_loader, dataset_train.phase_size)
        # learner.print_model_parameters()
        learner.before_validation()

        val_meter = validate(
            learner,
            test_loader,
            dataset_train.num_classes,
            desc=f"Phase {phase}",
        )
        sum_acc += val_meter.accuracy
        print(
            f"loss: {val_meter.loss:.4f}",
            f"acc@1: {val_meter.accuracy * 100:.3f}%",
            f"acc@5: {val_meter.accuracy5 * 100:.3f}%",
            f"f1-micro: {val_meter.f1_micro * 100:.3f}%",
            f"acc@avg: {sum_acc / (phase + 1) * 100:.3f}%",
            sep="    ",
        )
        print(
            phase,
            sum_acc / (phase + 1),
            val_meter.accuracy,
            val_meter.accuracy5,
            val_meter.f1_micro,
            val_meter.loss,
            file=log_file,
            sep=",",
        )
    log_file.close()

if __name__ == "__main__":
    main(load_args())





