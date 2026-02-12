import json
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader
from pathlib import Path
from src.data.datasets import Sen12Landslides


class Sen12LsDataModule(LightningDataModule):
    def __init__(
        self,
        root_dir,
        dataset_cfg,        
        batch_size,
        num_workers,
        train_transforms=None,
        val_transforms=None,
        test_transforms=None,
        pin_memory=True
    ):
        super().__init__()
        self.root = Path(root_dir)
        self.version = dataset_cfg.version
        self.task = dataset_cfg.task
        self.modality = dataset_cfg.modality.lower()
        self.use_dem = dataset_cfg.get("dem", False)
        self.min_date = dataset_cfg.get("min_date", "2015-12-03")
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.test_transforms = test_transforms
        self.pin_memory = pin_memory

    def _resolve_paths(self, splits, root):
        """Convert relative paths to absolute."""
        for split_list in splits.values():
            for entry in split_list:
                if self.modality in entry and not Path(entry[self.modality]).is_absolute():
                    entry[self.modality] = str(root / entry[self.modality])
        return splits
    
    def setup(self, stage=None):
        splits_file = self.root / "tasks" / self.task / self.version / self.modality / "splits.json"
        with open(splits_file, "r") as f:
            splits = json.load(f)
            
        dataset_root = self.root / "data" / f"data_{self.version}"
        splits = self._resolve_paths(splits, dataset_root)
        
        self.train_ds = Sen12Landslides(
            files=splits["train"],
            modality=self.modality,
            min_date=self.min_date,
            use_dem=self.use_dem,
            transforms=self.train_transforms,
        )
        self.val_ds = Sen12Landslides(
            files=splits["val"],
            modality=self.modality,
            min_date=self.min_date,
            use_dem=self.use_dem,
            transforms=self.val_transforms,
        )
        self.test_ds = Sen12Landslides(
            files=splits["test"],
            modality=self.modality,
            min_date=self.min_date,
            use_dem=self.use_dem,
            transforms=self.test_transforms,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds, 
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers, 
            pin_memory=self.pin_memory, 
            drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers, 
            pin_memory=self.pin_memory
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers, 
            pin_memory=self.pin_memory
        )