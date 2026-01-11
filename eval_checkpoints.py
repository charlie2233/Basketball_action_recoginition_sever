"""
Usage:
    python eval_checkpoints.py

Evaluates all specified checkpoints on the test split (same 10% split logic and seed as training)
and writes results to eval_results.json. Also prints a ranking by test accuracy (tie-break by macro F1).
"""

import argparse
import json
import os
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import models
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

from dataset import BasketballDataset


def build_model(num_classes: int, device: torch.device):
    # mirror train.py setup
    model = models.video.r2plus1d_18(pretrained=True, progress=True)
    for param in model.parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        for layer in ["layer3", "layer4", "fc"]:
            if layer in name:
                param.requires_grad = True
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes, bias=True)
    return model.to(device)


def load_checkpoint(model: torch.nn.Module, ckpt_path: Path, device: torch.device, num_classes: int):
    state = torch.load(str(ckpt_path), map_location=device)
    sd = state.get("state_dict", state)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"[warn] Checkpoint {ckpt_path.name} missing keys: {missing}, unexpected: {unexpected}")
    # hard fail if classifier shape mismatches
    fc_weight = model.fc.weight
    if fc_weight.shape[0] != num_classes:
        raise ValueError(
            f"Classifier head mismatch for {ckpt_path.name}: expected {num_classes} classes, "
            f"found {fc_weight.shape[0]}"
        )
    return model


def make_splits(annotation_path: str, batch_size: int = 8):
    dataset = BasketballDataset(
        annotation_dict=annotation_path,
        augmented_dict=None,
        augment=False,
    )
    N = len(dataset)
    test_n = round(0.1 * N)
    val_n = round(0.1 * N)
    train_n = N - test_n - val_n
    if train_n <= 0:
        raise ValueError("Train split non-positive; check dataset size.")

    train_subset, temp_subset = random_split(
        dataset,
        [train_n, test_n + val_n],
        generator=torch.Generator().manual_seed(1),
    )
    val_subset, test_subset = random_split(
        temp_subset,
        [val_n, test_n],
        generator=torch.Generator().manual_seed(1),
    )
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)
    return dataset, train_subset, val_subset, test_subset, (train_loader, val_loader, test_loader)


def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    all_preds: List[int] = []
    all_labels: List[int] = []
    with torch.no_grad():
        for idx, sample in enumerate(loader):
            inputs = sample["video"].to(device)
            labels = sample["action"].to(device)
            outputs = model(inputs)
            preds = outputs.argmax(1).cpu().tolist()
            gt = labels.argmax(1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(gt)

    acc = accuracy_score(all_labels, all_preds)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0
    )
    precision_pc, recall_pc, f1_pc, support_pc = precision_recall_fscore_support(
        all_labels, all_preds, average=None, zero_division=0
    )
    cm = confusion_matrix(all_labels, all_preds).tolist()

    per_class = []
    for i, (p, r, f1, s) in enumerate(zip(precision_pc, recall_pc, f1_pc, support_pc)):
        per_class.append(
            {"class": int(i), "precision": float(p), "recall": float(r), "f1": float(f1), "support": int(s)}
        )

    return {
        "accuracy": float(acc),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "per_class": per_class,
        "confusion_matrix": cm,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate checkpoints on test split")
    parser.add_argument("--print_split_counts", action="store_true", help="Print split sizes and sample ids")
    parser.add_argument("--sanity_print", action="store_true", help="Print label mapping and sample/pred sanity info")
    args_cli = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    annotation_path = "dataset/annotation_dict.json"
    labels_path = "dataset/labels_dict.json"
    checkpoints = [
        "model_checkpoints/r2plus1d_clean/r2plus1d_multiclass_3_0.0001.pt",
        "model_checkpoints/r2plus1d_clean/r2plus1d_multiclass_12_0.0001.pt",
        "model_checkpoints/r2plus1d_clean/r2plus1d_multiclass_13_0.0001.pt",
    ]

    dataset, train_subset, val_subset, test_subset, loaders = make_splits(annotation_path, batch_size=8)
    train_loader, val_loader, test_loader = loaders

    # labels mapping sanity
    labels_map = {}
    if Path(labels_path).exists():
        with open(labels_path) as f:
            raw = json.load(f)
        # normalize to id->name mapping
        try:
            # case: keys are ids (possibly str), values are names
            if all(str(k).isdigit() for k in raw.keys()):
                labels_map = {int(k): v for k, v in raw.items()}
            else:
                # case: keys are names, values are ids
                labels_map = {int(v): k for k, v in raw.items()}
        except Exception:
            labels_map = raw

    def unwrap_subset(subset):
        # resolve nested Subset -> base dataset and flat indices
        if not hasattr(subset, "dataset"):
            return subset, None
        if not hasattr(subset, "indices"):
            return subset, None
        base = subset.dataset
        idxs = list(subset.indices)
        while hasattr(base, "dataset") and hasattr(base, "indices"):
            idxs = [base.indices[i] for i in idxs]
            base = base.dataset
        return base, idxs

    def get_ids(subset):
        base, idxs = unwrap_subset(subset)
        if idxs is not None and hasattr(base, "video_list"):
            return [base.video_list[i][0] for i in idxs]
        if hasattr(base, "video_list"):
            return [x[0] for x in base.video_list]
        return []

    if args_cli.print_split_counts:
        train_ids = get_ids(train_subset)
        val_ids = get_ids(val_subset)
        test_ids = get_ids(test_subset)
        print(f"Total dataset: {len(dataset)}, train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")
        # overlap check
        overlap = set(train_ids) & set(val_ids) | set(train_ids) & set(test_ids) | set(val_ids) & set(test_ids)
        assert not overlap, f"Split overlap detected: {overlap}"
        print("First 5 train ids:", train_ids[:5])
        print("First 5 val ids:", val_ids[:5])
        print("First 5 test ids:", test_ids[:5])

    if args_cli.sanity_print:
        all_ids = get_ids(dataset)
        # label stats
        label_ids = [lbl for _, lbl in dataset.video_list] if hasattr(dataset, "video_list") else []
        print(f"num_classes (dataset): {len(set(label_ids))}, min_label={min(label_ids)}, max_label={max(label_ids)}")
        if labels_map:
            first_lbls = list(labels_map.items())[:10]
            print("labels_dict (first 10 id->name):", first_lbls)
        # random sample prints
        random.seed(42)
        sample_ids = random.sample(all_ids, min(10, len(all_ids)))
        print("Random 10 samples (id, label_id, label_name):")
        for vid in sample_ids:
            label_id = dict(dataset.video_list).get(vid)
            label_name = labels_map.get(label_id, None) if labels_map else None
            print((vid, label_id, label_name))

    results: Dict[str, dict] = {}
    for ckpt in checkpoints:
        ckpt_path = Path(ckpt)
        print(f"Evaluating {ckpt_path}...")
        num_classes = 10
        model = build_model(num_classes=num_classes, device=device)
        model = load_checkpoint(model, ckpt_path, device, num_classes)
        # optional: small batch prediction print
        if args_cli.sanity_print:
            model.eval()
            with torch.no_grad():
                for sample in test_loader:
                    vids = sample["video"].to(device)
                    labels = sample["action"].to(device)
                    outputs = model(vids)
                    preds = outputs.argmax(1).cpu().tolist()
                    gt = labels.argmax(1).cpu().tolist()
                    names_pred = [labels_map.get(p, p) for p in preds] if labels_map else preds
                    names_gt = [labels_map.get(g, g) for g in gt] if labels_map else gt
                    print("Sanity batch preds:", list(zip(gt, names_gt, preds, names_pred)))
                    break
        res = evaluate(model, test_loader, device)
        results[ckpt_path.name] = res
        print(
            f"{ckpt_path.name}: acc={res['accuracy']:.4f}, "
            f"macro_f1={res['f1_macro']:.4f}, macro_prec={res['precision_macro']:.4f}, macro_rec={res['recall_macro']:.4f}"
        )

    # rank by accuracy then macro f1
    ranked = sorted(
        results.items(),
        key=lambda kv: (kv[1]["f1_macro"], kv[1]["accuracy"]),
        reverse=True,
    )
    print("\nRanking by macro F1 (tie-break acc):")
    for i, (name, res) in enumerate(ranked, 1):
        print(f"{i}. {name} | macro_f1={res['f1_macro']:.4f}, acc={res['accuracy']:.4f}")

    with open("eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved results to eval_results.json")


if __name__ == "__main__":
    main()
