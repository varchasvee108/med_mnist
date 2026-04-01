from medmnist import PathMNIST
from torchvision import transforms


def get_transforms():
    return transforms.Compose([transforms.ToTensor()])


def get_dataset(split: str, data_dir: str):
    dataset = PathMNIST(
        split=split, root=data_dir, transform=get_transforms(), download=True
    )
    return dataset
