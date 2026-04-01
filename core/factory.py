import torch
from core.config import Config
from model.model import ViT
from transformers import get_scheduler


def create_model(config: Config) -> ViT:
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    model = ViT(config).to(device)
    return model


def create_optimizer(config: Config, model: torch.nn.Module) -> torch.optim.Optimizer:
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if (
            name.endswith("bias")
            or "norm" in name.lower()
            or "pos_embed" in name
            or "cls_token" in name
        ):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optim_groups = [
        {"params": decay_params, "weight_decay": config.training.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    return torch.optim.AdamW(
        optim_groups,
        lr=config.training.lr,
        betas=config.training.betas,
    )


def create_scheduler(
    config: Config, optimizer: torch.optim.Optimizer
) -> torch.optim.lr_scheduler._LRScheduler:
    return get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=config.training.warmup_steps,
        num_training_steps=config.training.max_steps,
    )
