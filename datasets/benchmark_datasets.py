"""
Benchmark datasets for evaluation (CIFAR100, Food101, etc.).
"""
from torchvision.datasets import CIFAR100, Food101, Flowers102, DTD, EuroSAT
from pathlib import Path

class BenchmarkDatasets:
    """Factory for loading standard benchmark datasets."""
    
    @staticmethod
    def get_cifar100(cache_dir: Path, transform, split='train'):
        """Get CIFAR100 dataset."""
        is_train = (split == 'train')
        return CIFAR100(
            root=cache_dir,  
            train=is_train,
            download=True,
            transform=transform
        )
    
    @staticmethod
    def get_food101(cache_dir: Path, transform, split='train'):
        """Get Food101 dataset."""
        return Food101(
            root=cache_dir, 
            split=split,
            download=True,
            transform=transform
        )
    
    @staticmethod
    def get_flowers102(cache_dir: Path, transform, split='train'):
        """Get Flowers102 dataset."""
        return Flowers102(
            root=cache_dir,  
            split=split,
            download=True,
            transform=transform
        )
    
    @staticmethod
    def get_dtd(cache_dir: Path, transform, split='train'):
        """Get Describable Textures Dataset."""
        return DTD(
            root=cache_dir, 
            split=split,
            download=True,
            transform=transform
        )
    
    @staticmethod
    def get_eurosat(cache_dir: Path, transform):
        """Get EuroSAT dataset."""
        return EuroSAT(
            root=cache_dir,
            download=True,
            transform=transform
        )
    
    @staticmethod
    def get_all_datasets(cache_dir: Path, transform, split='test'):
        """Get all benchmark datasets."""
        return {
            'CIFAR100': BenchmarkDatasets.get_cifar100(cache_dir, transform, split),
            'Food101': BenchmarkDatasets.get_food101(cache_dir, transform, split),
            'Flowers102': BenchmarkDatasets.get_flowers102(cache_dir, transform, split),
            'DTD': BenchmarkDatasets.get_dtd(cache_dir, transform, split),
            'EuroSAT': BenchmarkDatasets.get_eurosat(cache_dir, transform)
        }
