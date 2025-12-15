from torchvision.datasets import CIFAR100, Food101, Flowers102, DTD, EuroSAT
from config import DATA_ROOT
import templates

class DatasetFactory:
    """
    Factory class to load datasets and retrieve their specific configurations.
    """
    
    @staticmethod
    def get_zeroshot_config(transform):
        """
        Returns a dictionary of datasets configured for Zero-Shot evaluation.
        """
        return {
            "CIFAR100": {
                "dataset": CIFAR100(root=DATA_ROOT, download=True, train=False, transform=transform),
                "class_getter": lambda ds: templates.CIFAR100_CLASS_NAMES,
                "templates": templates.CIFAR100_TEMPLATES
            },
            "Food101": {
                "dataset": Food101(root=DATA_ROOT, download=True, split='test', transform=transform),
                "class_getter": lambda ds: templates.FOOD101_CLASS_NAMES,
                "templates": templates.FOOD101_TEMPLATES
            },
            "Flowers102": {
                "dataset": Flowers102(root=DATA_ROOT, download=True, split='test', transform=transform),
                "class_getter": lambda ds: templates.FLOWERS102_CLASS_NAMES,
                "templates": templates.FLOWERS102_TEMPLATES
            },
            "DescribableTextures": {
                "dataset": DTD(root=DATA_ROOT, download=True, split='test', transform=transform),
                "class_getter": lambda ds: templates.DESCRIBEABLETEXTURES_CLASS_NAMES,
                "templates": templates.DESCRIBEABLETEXTURES_TEMPLATES
            },
            "EuroSAT": {
                "dataset": EuroSAT(root=DATA_ROOT, download=True, transform=transform),
                "class_getter": lambda ds: templates.EUROSAT_CLASS_NAMES,
                "templates": templates.EUROSAT_TEMPLATES
            },
        }

    @staticmethod
    def get_linear_probe_datasets(transform):
        """
        Returns a dictionary containing 'train' and 'test' dataset objects for Linear Probe.
        """
        # Dictionary structure: Name -> {'train': dataset, 'test': dataset}
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
                # EuroSAT doesn't have standard train/test splits in Torchvision, 
                # using same set for demo or needs custom split logic.
                "train": EuroSAT(root=DATA_ROOT, download=True, transform=transform),
                "test": EuroSAT(root=DATA_ROOT, download=True, transform=transform),
            },
        }