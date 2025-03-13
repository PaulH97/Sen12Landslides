import argparse
import json
import logging
from pathlib import Path

import torch
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassF1Score, MulticlassPrecision, MulticlassRecall, MulticlassJaccardIndex

logging.basicConfig(level=logging.INFO)

def load_predictions(pred_file: Path):
    """
    Load 'preds' and 'masks' from a .pt file. 
    Each is typically [N, H, W] (class indices) or [N, ...] with some shape.
    """
    data = torch.load(pred_file, weights_only=True)
    return data["preds"], data["masks"]

def majority_vote_merge(*pred_tensors: torch.Tensor) -> torch.Tensor:
    """
    Merge multiple binary class-index predictions by majority vote.
    pred_tensors[i].shape => [N, H, W]
    Returns a [N, H, W] Tensor with the most frequent class (0 or 1) for each pixel.
    """
    stack = torch.stack(pred_tensors, dim=0)  # [num_preds, N, H, W]
    merged = torch.sum(stack, dim=0) >= (len(pred_tensors) / 2)
    return merged.long()

def compute_metrics(preds: torch.Tensor, masks: torch.Tensor, num_classes: int) -> dict:
    """
    Compute F1, Precision, Recall, and Jaccard for multiclass predictions.
    preds, masks => [N, H, W] with class indices in [0..num_classes-1].
    Returns a dict of metric_name -> value.
    """
    metric_collection = MetricCollection({
        "F1Score": MulticlassF1Score(num_classes=num_classes, average='macro'),
        "Precision": MulticlassPrecision(num_classes=num_classes, average='macro'),
        "Recall": MulticlassRecall(num_classes=num_classes, average='macro'),
        "Jaccard": MulticlassJaccardIndex(num_classes=num_classes, average='macro')
    })

    # Ensure everything is on CPU and correct type
    preds = preds.long().cpu()
    masks = masks.long().cpu()

    results = metric_collection(preds, masks)
    return {k: float(v.item()) for k, v in results.items()}

def main(exp_dir):

    root_dir = Path(exp_dir)

    models_with_preds = {}
    for data_dir in root_dir.iterdir():
        if data_dir.is_dir():
            for model_dir in data_dir.iterdir():
                if model_dir.is_dir():
                    model_name = model_dir.name
                    if model_name not in models_with_preds:
                        models_with_preds[model_name] = {}
                    pred_file = model_dir / "predictions" / "predictions.pt"
                    models_with_preds[model_name][data_dir.name] = pred_file

    model_metrics = {}

    for model, preds_files in models_with_preds.items():

        s1asc_preds, _ = load_predictions(preds_files["s1asc"])
        s1dsc_preds, _ = load_predictions(preds_files["s1dsc"])
        s2_preds, s2_masks = load_predictions(preds_files["s2"])

        s1_merged = majority_vote_merge(s1asc_preds, s1dsc_preds)
        s12_merged = majority_vote_merge(s1_merged, s2_preds)

        s1_metrics = compute_metrics(s1_merged, s2_masks, num_classes=2)
        s2_metrics = compute_metrics(s2_preds, s2_masks, num_classes=2)
        s12_metrics = compute_metrics(s12_merged, s2_masks, num_classes=2)

        model_metrics[model] = {"S1": s1_metrics, "S2": s2_metrics, "S12": s12_metrics}

    # Write out final JSON
    out_path = root_dir / "metrics.json"
    with open(out_path, "w") as f:
        json.dump(model_metrics, f, indent=4)
    logging.info(f"Results saved to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, required=True, help="Path to the root experiment directory (e.g. 'outputs/exp1/final').")
    args = parser.parse_args()
    main(args.exp_dir)
