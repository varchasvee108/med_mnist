from torch.utils.data import DataLoader
from data.dataset import get_dataset
from core.config import Config


def get_dataloader(config: Config):
    train_dataset = get_dataset(split="train", data_dir=config.data.data_dir)
    val_dataset = get_dataset(split="val", data_dir=config.data.data_dir)
    test_dataset = get_dataset(split="test", data_dir=config.data.data_dir)

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=True,
    )
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=True,
    )
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=True,
    )
    return train_dataloader, val_dataloader, test_dataloader
