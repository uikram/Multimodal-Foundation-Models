from torchvision.datasets import CIFAR100, Food101, Flowers102, DTD, EuroSAT
from config import DATA_ROOT
import templates
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode

class DatasetFactory:
    """
    Factory class to load datasets and retrieve their specific configurations.
    """
    
    @staticmethod
    def get_linear_probe_datasets(transform):
        """
        Returns a dictionary containing 'train' and 'test' dataset objects for Linear Probe.
        """
        return {
            "CIFAR100": {
                "train": CIFAR100(root=DATA_ROOT, download=True, train=True, transform=transform),
                "test": CIFAR100(root=DATA_ROOT, download=True, train=False, transform=transform),
            },
            "Food101": {
                "train": Food101(root=DATA_ROOT, download=True, split='train', transform=transform),
                "test": Food101(root=DATA_ROOT, download=True, split='test', transform=transform),
            },
            "Flowers102": {
                "train": Flowers102(root=DATA_ROOT, download=True, split='train', transform=transform),
                "test": Flowers102(root=DATA_ROOT, download=True, split='test', transform=transform),
            },
            "DTD": {
                "train": DTD(root=DATA_ROOT, download=True, split='train', transform=transform),
                "test": DTD(root=DATA_ROOT, download=True, split='test', transform=transform),
            },
            "EuroSAT": {
                "train": EuroSAT(root=DATA_ROOT, download=True, transform=transform),
                "test": EuroSAT(root=DATA_ROOT, download=True, transform=transform),
            },
        }