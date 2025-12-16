"""
Benchmark datasets for evaluation (CIFAR100, Food101, etc.).
"""

from torchvision.datasets import CIFAR100, Food101, Flowers102, DTD, EuroSAT
from pathlib import Path


class BenchmarkDatasets:
    """Factory for loading standard benchmark datasets."""
    
    # Class names
    CIFAR100_CLASSES = [
        'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
        'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
        'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
        'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
        'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
        'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
        'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
        'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
        'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
        'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea',
        'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider',
        'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank',
        'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip',
        'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
    ]
    
    @staticmethod
    def get_cifar100(data_root: Path, transform, split='train'):
        """Get CIFAR100 dataset."""
        is_train = (split == 'train')
        return CIFAR100(
            root=data_root,
            train=is_train,
            download=True,
            transform=transform
        )
    
    @staticmethod
    def get_food101(data_root: Path, transform, split='train'):
        """Get Food101 dataset."""
        return Food101(
            root=data_root,
            split=split,
            download=True,
            transform=transform
        )
    
    @staticmethod
    def get_flowers102(data_root: Path, transform, split='train'):
        """Get Flowers102 dataset."""
        return Flowers102(
            root=data_root,
            split=split,
            download=True,
            transform=transform
        )
    
    @staticmethod
    def get_dtd(data_root: Path, transform, split='train'):
        """Get Describable Textures Dataset."""
        return DTD(
            root=data_root,
            split=split,
            download=True,
            transform=transform
        )
    
    @staticmethod
    def get_eurosat(data_root: Path, transform):
        """Get EuroSAT dataset."""
        return EuroSAT(
            root=data_root,
            download=True,
            transform=transform
        )
    
    @staticmethod
    def get_all_datasets(data_root: Path, transform, split='test'):
        """Get all benchmark datasets."""
        return {
            'CIFAR100': BenchmarkDatasets.get_cifar100(data_root, transform, split),
            'Food101': BenchmarkDatasets.get_food101(data_root, transform, split),
            'Flowers102': BenchmarkDatasets.get_flowers102(data_root, transform, split),
            'DTD': BenchmarkDatasets.get_dtd(data_root, transform, split),
            'EuroSAT': BenchmarkDatasets.get_eurosat(data_root, transform)
        }
