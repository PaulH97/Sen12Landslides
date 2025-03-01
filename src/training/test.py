import torch
from pathlib import Path 
import yaml
import pytorch_lightning as pl
import json
import wandb
import gc
import argparse
import logging

from benchmarking.models.modelTask import ClassificationTask
from benchmarking.data_loading import get_dataloaders
from benchmarking.models import get_model

def test(model_config, iteration_idx):
    """Main testing function that handles predictions for different satellites and merges them."""
    try:
        model_config = Path(model_config)
        with open(model_config, 'r') as file:
            config = yaml.safe_load(file)
        model_name = model_config.stem

        # Get the iteration data
        base_dir = Path(config["DATA"]["base_dir"])
        exp_data_dir = Path(config["DATA"]["exp_data_dir"])
        satellite = config["DATA"]["satellite"]
        data_dir = exp_data_dir / satellite / f"i{iteration_idx}"
        data_paths_file = data_dir / "data_paths.json"
        norm_file = data_dir / "norm_data.json"

        # Load pre-existing output directories created by pretrain_setup.py
        model_dir = data_dir / model_name 
        train_dir = model_dir / "training"
        test_dir = model_dir / "test"

        if not train_dir.exists() or not test_dir.exists():
            raise FileNotFoundError(f"{train_dir} or {test_dir} does not exist. Make sure they are created during the pretraining setup.")

        # Initialize dataloaders for training, validation, and testing datasets
        train_dl, val_dl, test_dl = get_dataloaders(config, base_dir, data_paths_file, norm_file)

        print("Length of train_dl: ", len(train_dl))
        print("Length of val_dl: ", len(val_dl))
        print("Length of test_dl: ", len(test_dl))

        ckpt_path = train_dir / "checkpoints" / f"{model_name}-best.ckpt"
                    
        model = get_model(config)
        model = ClassificationTask.load_from_checkpoint(model=model, checkpoint_path=ckpt_path, config=config)

        trainer = pl.Trainer(accelerator="gpu" if torch.cuda.is_available() else "cpu", devices=1)

        if trainer.is_global_zero:
            # Evaluate test metrics (optional)
            test_metrics = trainer.test(model=model, dataloaders=test_dl)
            metrics_file = test_dir / "metrics.json"
            with open(metrics_file, "w") as f:
                json.dump(test_metrics, f, indent=4)
            print(f"Test metrics saved to {metrics_file}")

        #         # Get predictions using predict
        #         predictions = trainer.predict(model, test_dl)
        #         save_predictions(predictions, test_dir / "predictions")

        #         # Combine predictions and masks across batches
        #         preds = torch.cat([torch.sigmoid(batch["preds"]) for batch in predictions], dim=0)
        #         masks = torch.cat([batch["masks"] for batch in predictions], dim=0)

        #         # Save predictions and masks for the current satellite
        #         all_preds[sat_path.name] = preds
        #         all_masks[sat_path.name] = masks

        # # Create a folder for merged metrics; folder name depends on the DEM flag
        # merged_dir = exp_dir / "test" / ("with_DEM" if with_dem else "without_DEM") / model_name / f"i{iteration_idx}"
        # merged_dir.mkdir(parents=True, exist_ok=True)

        # # Merge S1 predictions (asc and dsc)
        # if (s1_asc_name in all_preds) and (s1_dsc_name in all_preds):
        #     preds_s1_list = [all_preds[s1_asc_name], all_preds[s1_dsc_name]]
        #     masks_s1_list = [all_masks[s1_asc_name], all_masks[s1_dsc_name]]
            
        #     df_s1 = test_thresholds_or(preds_s1_list, masks_s1_list)
        #     df_s1.to_csv(merged_dir / "S1_threshold_metrics.csv", index=False)
        #     print("Saved S1 threshold metrics.")

        # # Process S2 predictions
        # if s2_name in all_preds:
        #     preds_merge_list = [all_preds[s2_name]]
        #     masks_merge_list = [all_masks[s2_name]]
        #     df_s2 = test_thresholds_or(preds_merge_list, masks_merge_list)
        #     df_s2.to_csv(merged_dir / "S2_threshold_metrics.csv", index=False)
        #     print("Saved S2 threshold metrics.")

        #     if (s1_asc_name in all_preds) and (s1_dsc_name in all_preds):
        #         preds_merge_list = [all_preds[s1_asc_name], all_preds[s1_dsc_name], all_preds[s2_name]]
        #         masks_merge_list = [all_masks[s1_asc_name], all_masks[s1_dsc_name], all_masks[s2_name]]
        #         df_s1_s2 = test_thresholds_or(preds_merge_list, masks_merge_list)
        #         df_s1_s2.to_csv(merged_dir / "S1_S2_threshold_metrics.csv", index=False)
        #         print("Saved S1+S2 threshold metrics.")

        # print(f"Merged metrics saved to {merged_dir}")

    except Exception as e:
        logging.exception(f"Test crashed on iteration {iteration_idx} with error: {e}")
    finally:
        wandb.finish()
        torch.cuda.empty_cache()  # Clear GPU memory after each iteration
        gc.collect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model with specific iteration index.")
    parser.add_argument("-c", "--model_config", required=True, help="Model config file")
    parser.add_argument("-i", "--iteration", type=int, required=True, help="Iteration index")
    args = parser.parse_args()
    test(args.model_config,args.iteration)

# salloc --account=pn76xa-c --cpus-per-task=8 --ntasks-per-node=1 --gres=gpu:1 --partition=hpda2_compute_gpu --time=03:00:00