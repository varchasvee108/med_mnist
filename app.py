import torch
import gradio as gr
import random
from torchvision import transforms

from core.config import Config
from model.model import ViT
from data.dataset import get_dataset


CONFIG_PATH = "config/config.toml"
MODEL_PATH = "./outputs/best_model.pth"

LABELS = [
    "adipose",
    "background",
    "debris",
    "lymphocytes",
    "mucus",
    "smooth muscle",
    "normal colon mucosa",
    "cancer-associated stroma",
    "colorectal adenocarcinoma epithelium",
]


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


config = Config.load_config(CONFIG_PATH)

model = ViT(config).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()


test_dataset = get_dataset(split="test", data_dir="./data")


def get_random_sample():
    idx = random.randint(0, len(test_dataset) - 1)

    image, label = test_dataset[idx]

    # convert to PIL for display
    pil_image = transforms.ToPILImage()(image)

    x = image.unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=-1)

        pred = probs.argmax(dim=1).item()
        confidence = probs[0][pred].item()

    return (
        pil_image,
        f"{LABELS[pred]} ({confidence * 100:.2f}%)",
        f"{LABELS[label.item()]}",
    )


with gr.Blocks() as demo:
    gr.Markdown("## Medical Image Classification (PathMNIST ViT)")

    with gr.Row():
        image_output = gr.Image(label="Image")
        pred_output = gr.Textbox(label="Prediction + Confidence")
        gt_output = gr.Textbox(label="Ground Truth")

    btn = gr.Button("Show Random Sample")

    btn.click(
        fn=get_random_sample,
        inputs=[],
        outputs=[image_output, pred_output, gt_output],
    )

demo.launch()
