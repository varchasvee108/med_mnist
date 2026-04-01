import torch
from core.config import Config
from core.factory import create_model
from data.dataloader import get_dataloader


def infer():
    config = Config.load_config("config/config.toml")

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    model = create_model(config).to(device)
    model.load_state_dict(torch.load("outputs/best_model.pth", map_location=device))
    model.eval()
    _, _, test_loader = get_dataloader(config)

    print(f"Running inference on {device}...")

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device).squeeze().long()

            logits = model(images)
            preds = logits.argmax(dim=1)

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    accuracy = (all_preds == all_labels).float().mean().item() * 100

    print(f"Test Accuracy: {accuracy:.2f}%")

    print("\nSample predictions:")
    for i in range(10):
        print(f"Pred: {all_preds[i].item()} | Label: {all_labels[i].item()}")


if __name__ == "__main__":
    infer()
