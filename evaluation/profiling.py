"""
Advanced profiling utilities for model performance analysis.
"""

import time
import torch
import psutil
import numpy as np
from typing import Dict, List, Callable, Any
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


class EfficiencyProfiler:
    """Profiler specifically designed for Table 5 in the paper."""
    
    def __init__(self, device='cuda'):
        self.device = device

    def profile_model(self, model, input_shape=(1, 3, 224, 224), num_iterations=100):
        """
        Measures Latency, Throughput, Peak Memory, and FLOPs (estimated).
        """
        model.eval()
        model.to(self.device)
        
        # Generate dummy input
        dummy_input = torch.randn(input_shape).to(self.device)
        
        # 1. Warmup (crucial for CUDA kernels)
        print("  Warmup...")
        with torch.no_grad():
            for _ in range(10):
                if hasattr(model, 'encode_image'):
                    _ = model.encode_image(dummy_input)
                else:
                    _ = model(dummy_input)
        
        # 2. Measure Latency
        print(f"  Profiling Latency ({num_iterations} runs)...")
        latencies = []
        with torch.no_grad():
            for _ in range(num_iterations):
                if self.device == 'cuda' and torch.cuda.is_available():
                    torch.cuda.synchronize()
                start_time = time.time()
                
                if hasattr(model, 'encode_image'):
                    _ = model.encode_image(dummy_input)
                else:
                    _ = model(dummy_input)
                
                if self.device == 'cuda' and torch.cuda.is_available():
                    torch.cuda.synchronize()
                latencies.append((time.time() - start_time) * 1000) # ms

        # 3. Measure PEAK Memory (Critical for Wearable Robotics)
        print("  Profiling Peak Memory...")
        if self.device == 'cuda' and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            with torch.no_grad():
                if hasattr(model, 'encode_image'):
                    _ = model.encode_image(dummy_input)
                else:
                    _ = model(dummy_input)
            peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2) # MB
        else:
            peak_memory = 0.0

        # 4. Parameter Counts
        params = model.count_parameters() if hasattr(model, 'count_parameters') else {}
        total_params = params.get('total', 0) / 1e6 # Million
        trainable_params = params.get('trainable', 0) / 1e6 # Million

        results = {
            "latency_ms": np.mean(latencies),
            "latency_std": np.std(latencies),
            "throughput": 1000 / np.mean(latencies),
            "peak_memory_mb": peak_memory,
            "params_total_M": total_params,
            "params_trainable_M": trainable_params
        }
        return results


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