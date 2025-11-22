import gc
import logging
import traceback
from pathlib import Path
import hydra
import torch
from hydra.utils import instantiate
from lightning.pytorch import seed_everything
from hydra.core.hydra_config import HydraConfig
from torchinfo import summary

from src.utils.helpers import run_sanity_check

@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg):
    try:
        torch.autograd.set_detect_anomaly(True)
        torch.set_float32_matmul_precision('high')
        # torch.use_deterministic_algorithms(True, warn_only=False)
        seed_everything(cfg.seed, workers=True)
    
        datamodule = instantiate(cfg.datamodule)
        model = instantiate(cfg.model.instance, _convert_="all")
        lit_module = instantiate(cfg.module, net=model, _convert_="all", _recursive_=False)

        if cfg.get("sanity_check", False):
            output_dir = Path(HydraConfig.get().runtime.output_dir) / "sanity_check"
            run_sanity_check(datamodule, output_dir, num_samples=5)
        # logging.info(f"Model summary: {summary(model)}")

        trainer = instantiate(cfg.trainer, _convert_="all")
        trainer.fit(lit_module, datamodule=datamodule)
        trainer.test(lit_module, datamodule=datamodule)

    except Exception as e:
        traceback.print_exc()
        logging.error(f"Training crashed: {e}")
    finally:
        torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    main()