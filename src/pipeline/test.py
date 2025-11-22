import hydra
import torch
import gc
import logging
import traceback
from hydra.utils import instantiate
from lightning import seed_everything


@hydra.main(config_path="../../../configs", config_name="config", version_base=None)
def main(cfg):
    try:
        torch.set_float32_matmul_precision('high')
        seed_everything(cfg.seed, workers=True)

        datamodule = instantiate(cfg.datamodule)
        datamodule.setup(stage="test")
        
        model = instantiate(cfg.model.instance, _convert_="all")
        lit_module = instantiate(cfg.module, net=model, _convert_="all", _recursive_=False)
        trainer = instantiate(cfg.trainer, logger=False, _convert_="all")

        ckpt_path = cfg.get("ckpt_path", None)
        
        trainer.test(lit_module, datamodule=datamodule, ckpt_path=ckpt_path)

    except Exception as e:
        traceback.print_exc()
        logging.error(f"Testing crashed: {e}")
    finally:
        torch.cuda.empty_cache()
        gc.collect()

        
if __name__ == "__main__":
    main()