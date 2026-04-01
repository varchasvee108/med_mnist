import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
import wandb
from pathlib import Path

from core.config import Config
from core.factory import create_model, create_optimizer, create_scheduler
from data.dataloader import get_dataloader


def train():
    config = Config.load_config("config/config.toml")

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    use_amp = device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)

    wandb.init(
        project="vit-medmnist",
        config={
            "lr": config.training.lr,
            "batch_size": config.training.batch_size,
            "max_steps": config.training.max_steps,
        },
    )

    save_dir = Path(config.logging.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, _ = get_dataloader(config)
    train_iter = iter(train_loader)

    model = create_model(config).to(device)
    optimizer = create_optimizer(config, model)
    scheduler = create_scheduler(config, optimizer)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0

    print(f"Training for {config.training.max_steps} steps on {device}...")

    for step in range(config.training.max_steps):
        model.train()

        try:
            images, labels = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            images, labels = next(train_iter)

        images = images.to(device)
        labels = labels.to(device).squeeze().long()

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        # ---- Logging ----
        if step % config.logging.log_interval == 0:
            lr = optimizer.param_groups[0]["lr"]

            wandb.log(
                {
                    "train/loss": loss.item(),
                    "train/lr": lr,
                    "step": step,
                }
            )

            print(f"Step {step} | Loss: {loss.item():.4f}")

        # ---- validation ----
        if step % (len(train_loader)) == 0 and step > 0:
            val_acc = evaluate(model, val_loader, device)

            wandb.log(
                {
                    "val/accuracy": val_acc,
                    "step": step,
                }
            )

            print(f"Step {step} | Val Acc: {val_acc:.2f}%")

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), save_dir / "best_model.pth")

    torch.save(model.state_dict(), save_dir / "final_model.pth")
    wandb.finish()


@torch.no_grad()
def evaluate(model, loader, device, max_batches=50):
    model.eval()
    correct, total = 0, 0

    for i, (images, labels) in enumerate(loader):
        if i >= max_batches:
            break
        images = images.to(device)
        labels = labels.to(device).squeeze().long()

        logits = model(images)
        preds = logits.argmax(dim=1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return (correct / total) * 100


if __name__ == "__main__":
    train()
