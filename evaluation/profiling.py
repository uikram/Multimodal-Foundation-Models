"""
Advanced profiling utilities for model performance analysis.
"""

import time
import torch
import psutil
import numpy as np
from typing import Dict, List, Callable
from dataclasses import dataclass


@dataclass
class ProfileResults:
    """Container for profiling results."""
    avg_time_ms: float
    std_time_ms: float
    min_time_ms: float
    max_time_ms: float
    throughput: float  # samples/second
    memory_allocated_mb: float
    memory_reserved_mb: float


class ModelProfiler:
    """Advanced profiler for model performance analysis."""
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        
    def profile_inference(
        self,
        input_generator: Callable,
        num_iterations: int = 100,
        warmup_iterations: int = 10
    ) -> ProfileResults:
        """
        Profile model inference performance.
        
        Args:
            input_generator: Function that generates model inputs
            num_iterations: Number of iterations to measure
            warmup_iterations: Number of warmup iterations (not measured)
        
        Returns:
            ProfileResults with timing and memory statistics
        """
        self.model.eval()
        times = []
        
        # Warmup
        print(f"Warming up ({warmup_iterations} iterations)...")
        with torch.no_grad():
            for _ in range(warmup_iterations):
                inputs = input_generator()
                _ = self._forward(inputs)
        
        # Profile
        print(f"Profiling ({num_iterations} iterations)...")
        with torch.no_grad():
            for _ in range(num_iterations):
                inputs = input_generator()
                
                # Synchronize GPU
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                start = time.perf_counter()
                _ = self._forward(inputs)
                
                # Synchronize GPU
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                elapsed = (time.perf_counter() - start) * 1000  # ms
                times.append(elapsed)
        
        # Get memory stats
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024 / 1024
            memory_reserved = torch.cuda.memory_reserved() / 1024 / 1024
        else:
            memory_allocated = 0
            memory_reserved = 0
        
        # Calculate statistics
        times = np.array(times)
        avg_time = float(np.mean(times))
        throughput = 1000.0 / avg_time  # samples per second
        
        return ProfileResults(
            avg_time_ms=avg_time,
            std_time_ms=float(np.std(times)),
            min_time_ms=float(np.min(times)),
            max_time_ms=float(np.max(times)),
            throughput=throughput,
            memory_allocated_mb=memory_allocated,
            memory_reserved_mb=memory_reserved
        )
    
    def _forward(self, inputs):
        """Forward pass handling different input types."""
        if isinstance(inputs, dict):
            return self.model(**inputs)
        elif isinstance(inputs, (list, tuple)):
            return self.model(*inputs)
        else:
            return self.model(inputs)
    
    def profile_batch_sizes(
        self,
        input_generator_factory: Callable[[int], Callable],
        batch_sizes: List[int] = [1, 2, 4, 8, 16, 32, 64, 128],
        num_iterations: int = 50
    ) -> Dict[int, ProfileResults]:
        """
        Profile model across different batch sizes.
        
        Args:
            input_generator_factory: Function that takes batch_size and returns input_generator
            batch_sizes: List of batch sizes to test
            num_iterations: Number of iterations per batch size
        
        Returns:
            Dictionary mapping batch_size -> ProfileResults
        """
        results = {}
        
        for batch_size in batch_sizes:
            print(f"\nProfiling batch size: {batch_size}")
            try:
                input_gen = input_generator_factory(batch_size)
                result = self.profile_inference(input_gen, num_iterations)
                results[batch_size] = result
                
                print(f"  Avg time: {result.avg_time_ms:.2f}ms")
                print(f"  Throughput: {result.throughput * batch_size:.2f} samples/sec")
                print(f"  Memory: {result.memory_allocated_mb:.2f}MB")
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"  ✗ Out of memory at batch size {batch_size}")
                    break
                else:
                    raise
        
        return results
    
    def compare_models(
        self,
        models: Dict[str, torch.nn.Module],
        input_generator: Callable,
        num_iterations: int = 100
    ) -> Dict[str, ProfileResults]:
        """
        Compare performance across multiple models.
        
        Args:
            models: Dictionary of model_name -> model
            input_generator: Function that generates inputs
            num_iterations: Number of iterations per model
        
        Returns:
            Dictionary mapping model_name -> ProfileResults
        """
        results = {}
        
        for name, model in models.items():
            print(f"\nProfiling {name}...")
            profiler = ModelProfiler(model, self.device)
            result = profiler.profile_inference(input_generator, num_iterations)
            results[name] = result
            
            print(f"  Avg time: {result.avg_time_ms:.2f}ms")
            print(f"  Throughput: {result.throughput:.2f} samples/sec")
        
        return results


class MemoryProfiler:
    """Memory usage profiler."""
    
    @staticmethod
    def get_gpu_memory() -> Dict[str, float]:
        """Get current GPU memory usage."""
        if not torch.cuda.is_available():
            return {'allocated_mb': 0, 'reserved_mb': 0, 'free_mb': 0}
        
        allocated = torch.cuda.memory_allocated() / 1024 / 1024
        reserved = torch.cuda.memory_reserved() / 1024 / 1024
        total = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
        free = total - allocated
        
        return {
            'allocated_mb': allocated,
            'reserved_mb': reserved,
            'free_mb': free,
            'total_mb': total
        }
    
    @staticmethod
    def get_cpu_memory() -> Dict[str, float]:
        """Get current CPU memory usage."""
        process = psutil.Process()
        mem_info = process.memory_info()
        
        return {
            'rss_mb': mem_info.rss / 1024 / 1024,  # Resident Set Size
            'vms_mb': mem_info.vms / 1024 / 1024,  # Virtual Memory Size
        }
    
    @staticmethod
    def print_memory_summary():
        """Print formatted memory summary."""
        gpu_mem = MemoryProfiler.get_gpu_memory()
        cpu_mem = MemoryProfiler.get_cpu_memory()
        
        print("\n" + "="*60)
        print("MEMORY SUMMARY")
        print("="*60)
        
        if gpu_mem['allocated_mb'] > 0:
            print("\nGPU Memory:")
            print(f"  Allocated: {gpu_mem['allocated_mb']:.2f} MB")
            print(f"  Reserved:  {gpu_mem['reserved_mb']:.2f} MB")
            print(f"  Free:      {gpu_mem['free_mb']:.2f} MB")
            print(f"  Total:     {gpu_mem['total_mb']:.2f} MB")
        else:
            print("\nGPU: Not available")
        
        print("\nCPU Memory:")
        print(f"  RSS: {cpu_mem['rss_mb']:.2f} MB")
        print(f"  VMS: {cpu_mem['vms_mb']:.2f} MB")
        print("="*60 + "\n")


# Example usage functions

def example_clip_profiling():
    """Example: Profile CLIP model."""
    from models.clip_baseline import CLIPBaseline
    from utils.config import CLIPConfig
    
    config = CLIPConfig()
    model = CLIPBaseline(config)
    profiler = ModelProfiler(model, config.device)
    
    # Define input generator
    def input_gen():
        dummy_image = torch.randn(1, 3, 224, 224).to(config.device)
        return (dummy_image,)
    
    # Profile
    results = profiler.profile_inference(input_gen, num_iterations=100)
    
    print(f"\nAverage inference time: {results.avg_time_ms:.2f}ms")
    print(f"Throughput: {results.throughput:.2f} samples/sec")
    print(f"Memory allocated: {results.memory_allocated_mb:.2f}MB")
