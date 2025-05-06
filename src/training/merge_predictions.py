import argparse
import json
import logging
import statistics
from pathlib import Path

import torch
from joblib import Parallel, delayed
from torchmetrics import MetricCollection
from torchmetrics.classification import (MulticlassF1Score,
                                         MulticlassJaccardIndex,
                                         MulticlassPrecision, MulticlassRecall)

logging.basicConfig(level=logging.INFO)


def load_predictions(pred_file):
    """
    Load predictions from a file.

    Parameters
    ----------
    pred_file : Path
        Path to the prediction file.

    Returns
    -------
    tuple
        A tuple containing:
        - preds: The predictions loaded from the file.
        - masks: The masks loaded from the file.
    """
    data = torch.load(pred_file, weights_only=True)
    return data["preds"], data["masks"]


def majority_vote_merge(*pred_tensors):
    """
    Perform majority voting to merge multiple binary prediction tensors.

    This function takes multiple binary prediction tensors and performs a majority vote
    to create a single merged prediction tensor. A pixel in the output is set to 1 if
    at least half of the input tensors have a 1 at that position, otherwise it's set to 0.

    Args:
        *pred_tensors (torch.Tensor): Variable number of prediction tensors with shape [N, H, W],
                                     where N is the batch size, H is height, and W is width.
                                     Values should be binary (0 or 1).

    Returns:
        torch.Tensor: Merged prediction tensor with shape [N, H, W] and values 0 or 1.
                     A pixel is set to 1 if at least half of the input tensors have
                     a 1 at that position.
    """
    stack = torch.stack(pred_tensors, dim=0)  # [num_preds, N, H, W]
    merged = torch.sum(stack, dim=0) >= (len(pred_tensors) / 2)
    return merged.long()


def compute_metrics(preds, masks, num_classes):
    """
    Compute evaluation metrics for multiclass predictions.

    This function calculates several metrics for evaluating the performance of
    multiclass segmentation predictions, including F1 Score, Precision, Recall,
    and Jaccard Index (IoU), all using macro averaging.

    Args:
        preds (torch.Tensor): Prediction tensor containing predicted class indices.
        masks (torch.Tensor): Ground truth tensor containing target class indices.
        num_classes (int): Number of classes in the segmentation task.

    Returns:
        dict: Dictionary containing the computed metrics with values rounded to
              2 decimal places. Keys include 'F1Score', 'Precision', 'Recall',
              and 'Jaccard'.
    """
    metric_collection = MetricCollection(
        {
            "F1Score": MulticlassF1Score(num_classes=num_classes, average="macro"),
            "Precision": MulticlassPrecision(num_classes=num_classes, average="macro"),
            "Recall": MulticlassRecall(num_classes=num_classes, average="macro"),
            "Jaccard": MulticlassJaccardIndex(num_classes=num_classes, average="macro"),
        }
    )
    preds = preds.long().cpu()
    masks = masks.long().cpu()

    results = metric_collection(preds, masks)
    return {k: round(float(v.item()), 2) for k, v in results.items()}


def process_single_run(run_idx, preds_files):
    """
    Process a single model run by merging predictions from different sensors and computing performance metrics.

    This function loads predictions from S1 ascending, S1 descending, and S2 sensors for a specific run,
    performs majority voting to merge the predictions, and computes evaluation metrics for each
    sensor type and combination.

    Parameters
    ----------
    run_idx : int
        Index of the run to process
    preds_files : dict
        Dictionary containing file paths for predictions from different sensors.
        Expected keys: 's1asc', 's1dsc', 's2', each containing a list of file paths.

    Returns
    -------
    dict
        Dictionary containing computed metrics for different sensor combinations:
        - 'S1': Metrics for merged S1 ascending and descending predictions
        - 'S2': Metrics for S2 predictions
        - 'S12': Metrics for merged S1 and S2 predictions
    """
    s1asc_preds, _ = load_predictions(preds_files["s1asc"][run_idx])
    s1dsc_preds, _ = load_predictions(preds_files["s1dsc"][run_idx])
    s2_preds, s2_masks = load_predictions(preds_files["s2"][run_idx])

    s1_merged = majority_vote_merge(s1asc_preds, s1dsc_preds)
    s12_merged = majority_vote_merge(s1_merged, s2_preds)

    s1_metrics = compute_metrics(s1_merged, s2_masks, num_classes=2)
    s2_metrics = compute_metrics(s2_preds, s2_masks, num_classes=2)
    s12_metrics = compute_metrics(s12_merged, s2_masks, num_classes=2)

    return {"S1": s1_metrics, "S2": s2_metrics, "S12": s12_metrics}


def main(exp_dir, n_jobs=3):
    """
    Process prediction files from multiple models and data sources, compute metrics, and save results.

    This function:
    1. Scans the experiment directory for model predictions across different data sources
    2. Processes predictions from 3 runs for each model using parallel execution
    3. Aggregates metrics by computing mean and standard deviation across runs
    4. Saves all metrics to a JSON file in the experiment root directory

    Parameters
    ----------
    exp_dir : str or Path
        Path to the experiment directory containing model predictions
    n_jobs : int, default=3
        Number of parallel jobs to run when processing predictions

    Notes
    -----
    The function expects a specific directory structure:
    exp_dir/
      └── data_source_1/
          └── model_name_1/
              └── predictions/
                  └── predictions_*.pt
      └── data_source_2/
          └── model_name_1/
              └── predictions/
                  └── predictions_*.pt

    Results are saved to exp_dir/metrics.json
    """

    root_dir = Path(exp_dir)
    models_with_preds = {}
    for data_dir in root_dir.iterdir():
        if data_dir.is_dir():
            for model_dir in data_dir.iterdir():
                if model_dir.is_dir():
                    model_name = model_dir.name
                    models_with_preds.setdefault(model_name, {})
                    preds_dir = model_dir / "predictions"
                    pred_files = sorted(preds_dir.glob("predictions_*.pt"))
                    models_with_preds[model_name][data_dir.name] = pred_files

    model_metrics = {}

    for model, preds_files in models_with_preds.items():
        logging.info(f"Processing model: {model}")

        metrics_list = Parallel(n_jobs=n_jobs)(
            delayed(process_single_run)(model, i, preds_files) for i in [0, 1, 2]
        )
        model_metrics[model] = metrics_list

    aggregated_metrics = {}

    for model, runs in model_metrics.items():
        model_aggregated = {}
        for method in ["S1", "S2", "S12"]:
            metrics_names = runs[0][method].keys()
            method_agg = {}
            for metric in metrics_names:
                scores = [run[method][metric] for run in runs]
                method_agg[metric] = {
                    "mean": round(statistics.mean(scores), 2),
                    "std": round(statistics.stdev(scores) if len(scores) > 1 else 0, 2),
                }
            model_aggregated[method] = method_agg
        aggregated_metrics[model] = model_aggregated

    model_metrics["aggregated_metrics"] = aggregated_metrics

    out_path = root_dir / "metrics.json"
    with open(out_path, "w") as f:
        json.dump(model_metrics, f, indent=4)

    logging.info(f"Results saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp_dir",
        type=str,
        required=True,
        help="Path to the root experiment directory (e.g. 'outputs/exp1/final').",
    )
    args = parser.parse_args()
    main(args.exp_dir)
