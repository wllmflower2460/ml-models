#!/usr/bin/env python3
"""
GPU Test Script - Verify CUDA setup and basic ML functionality
"""
import torch
import time

def test_gpu():
    print("=== GPU Test Results ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Device count: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name()}")
        
        # Memory info
        print(f"Memory allocated: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
        print(f"Memory reserved: {torch.cuda.memory_reserved() / 1024**2:.1f} MB")
        
        # Simple computation test
        print("\n=== Performance Test ===")
        size = 5000
        
        # CPU test
        cpu_tensor = torch.randn(size, size)
        start_time = time.time()
        cpu_result = torch.matmul(cpu_tensor, cpu_tensor)
        cpu_time = time.time() - start_time
        print(f"CPU matrix multiplication ({size}x{size}): {cpu_time:.3f}s")
        
        # GPU test
        gpu_tensor = torch.randn(size, size, device='cuda')
        torch.cuda.synchronize()
        start_time = time.time()
        gpu_result = torch.matmul(gpu_tensor, gpu_tensor)
        torch.cuda.synchronize()
        gpu_time = time.time() - start_time
        print(f"GPU matrix multiplication ({size}x{size}): {gpu_time:.3f}s")
        print(f"GPU speedup: {cpu_time/gpu_time:.1f}x")
        
    else:
        print("CUDA is not available. Check your installation.")

if __name__ == "__main__":
    test_gpu()