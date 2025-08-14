"""
Model utility functions for WavePose.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Union
import numpy as np


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count the number of parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary containing parameter counts
    """
    total_params = 0
    trainable_params = 0
    
    for param in model.parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': total_params - trainable_params
    }


def get_model_size(model: nn.Module, precision: str = 'fp32') -> Dict[str, Union[int, float]]:
    """
    Calculate the model size in memory.
    
    Args:
        model: PyTorch model
        precision: Model precision ('fp32', 'fp16', 'int8')
        
    Returns:
        Dictionary containing size information
    """
    # Count parameters
    param_counts = count_parameters(model)
    
    # Calculate size based on precision
    if precision == 'fp32':
        bytes_per_param = 4
    elif precision == 'fp16':
        bytes_per_param = 2
    elif precision == 'int8':
        bytes_per_param = 1
    else:
        raise ValueError(f"Unknown precision: {precision}")
    
    # Calculate sizes
    total_size_bytes = param_counts['total'] * bytes_per_param
    trainable_size_bytes = param_counts['trainable'] * bytes_per_param
    
    # Convert to different units
    total_size_mb = total_size_bytes / (1024 * 1024)
    trainable_size_mb = trainable_size_bytes / (1024 * 1024)
    
    return {
        'total_params': param_counts['total'],
        'trainable_params': param_counts['trainable'],
        'total_size_mb': total_size_mb,
        'trainable_size_mb': trainable_size_mb,
        'precision': precision
    }


def print_model_summary(model: nn.Module, input_size: Tuple[int, ...] = None) -> None:
    """
    Print a detailed summary of the model.
    
    Args:
        model: PyTorch model
        input_size: Input tensor size for calculating FLOPs
    """
    print("=" * 60)
    print("MODEL SUMMARY")
    print("=" * 60)
    
    # Parameter counts
    param_counts = count_parameters(model)
    print(f"Parameters:")
    print(f"  Total: {param_counts['total']:,}")
    print(f"  Trainable: {param_counts['trainable']:,}")
    print(f"  Non-trainable: {param_counts['non_trainable']:,}")
    
    # Model size
    model_size = get_model_size(model)
    print(f"Model Size:")
    print(f"  Total: {model_size['total_size_mb']:.2f} MB")
    print(f"  Trainable: {model_size['trainable_size_mb']:.2f} MB")
    print(f"  Precision: {model_size['precision']}")
    
    # Layer information
    print(f"\nLayer Information:")
    total_params = 0
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf module
            params = sum(p.numel() for p in module.parameters())
            if params > 0:
                print(f"  {name}: {params:,} parameters")
                total_params += params
    
    print(f"  Total layers with parameters: {total_params:,}")
    
    # FLOPs estimation (if input size provided)
    if input_size:
        try:
            flops = estimate_flops(model, input_size)
            print(f"\nFLOPs Estimation:")
            print(f"  Input size: {input_size}")
            print(f"  Estimated FLOPs: {flops:,}")
        except Exception as e:
            print(f"\nFLOPs estimation failed: {e}")
    
    print("=" * 60)


def estimate_flops(model: nn.Module, input_size: Tuple[int, ...]) -> int:
    """
    Estimate the number of FLOPs for a model.
    
    Args:
        model: PyTorch model
        input_size: Input tensor size
        
    Returns:
        Estimated number of FLOPs
    """
    # This is a simplified FLOP estimation
    # In practice, you'd use tools like thop or ptflops
    
    def count_conv2d_flops(module, input_shape):
        output_shape = list(module.output_shape)
        kernel_ops = module.kernel_size[0] * module.kernel_size[1] * module.in_channels // module.groups
        flops = kernel_ops * output_shape[0] * output_shape[1] * module.out_channels
        return flops
    
    def count_linear_flops(module, input_shape):
        return module.in_features * module.out_features
    
    total_flops = 0
    
    # Create a dummy input
    dummy_input = torch.randn(input_size)
    
    # Register hooks to capture output shapes
    output_shapes = {}
    
    def hook_fn(module, input, output):
        if isinstance(output, torch.Tensor):
            output_shapes[module] = output.shape
        else:
            output_shapes[module] = [o.shape for o in output]
    
    hooks = []
    for name, module in model.named_modules():
        hook = module.register_forward_hook(hook_fn)
        hooks.append(hook)
    
    # Forward pass
    with torch.no_grad():
        model(dummy_input)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Calculate FLOPs
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            if hasattr(module, 'output_shape'):
                flops = count_conv2d_flops(module, input_size)
                total_flops += flops
        elif isinstance(module, nn.Linear):
            flops = count_linear_flops(module, input_size)
            total_flops += flops
    
    return total_flops


def freeze_layers(model: nn.Module, layer_names: List[str]) -> None:
    """
    Freeze specific layers in the model.
    
    Args:
        model: PyTorch model
        layer_names: List of layer names to freeze
    """
    for name, param in model.named_parameters():
        for layer_name in layer_names:
            if layer_name in name:
                param.requires_grad = False
                print(f"Frozen layer: {name}")


def unfreeze_layers(model: nn.Module, layer_names: List[str]) -> None:
    """
    Unfreeze specific layers in the model.
    
    Args:
        model: PyTorch model
        layer_names: List of layer names to unfreeze
    """
    for name, param in model.named_parameters():
        for layer_name in layer_names:
            if layer_name in name:
                param.requires_grad = True
                print(f"Unfrozen layer: {name}")


def get_layer_outputs(model: nn.Module, input_tensor: torch.Tensor, 
                     layer_names: List[str]) -> Dict[str, torch.Tensor]:
    """
    Get intermediate layer outputs.
    
    Args:
        model: PyTorch model
        input_tensor: Input tensor
        layer_names: List of layer names to capture outputs from
        
    Returns:
        Dictionary containing layer outputs
    """
    outputs = {}
    
    def hook_fn(name):
        def hook(module, input, output):
            outputs[name] = output
        return hook
    
    hooks = []
    for name, module in model.named_modules():
        if name in layer_names:
            hook = module.register_forward_hook(hook_fn(name))
            hooks.append(hook)
    
    # Forward pass
    with torch.no_grad():
        model(input_tensor)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    return outputs


def calculate_gradient_norm(model: nn.Module) -> float:
    """
    Calculate the L2 norm of gradients.
    
    Args:
        model: PyTorch model
        
    Returns:
        Gradient L2 norm
    """
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm


def save_model_architecture(model: nn.Module, save_path: str) -> None:
    """
    Save the model architecture to a text file.
    
    Args:
        model: PyTorch model
        save_path: Path to save the architecture
    """
    with open(save_path, 'w') as f:
        f.write(str(model))
        
        # Add parameter information
        f.write("\n\n" + "="*60 + "\n")
        f.write("PARAMETER INFORMATION\n")
        f.write("="*60 + "\n")
        
        param_counts = count_parameters(model)
        f.write(f"Total parameters: {param_counts['total']:,}\n")
        f.write(f"Trainable parameters: {param_counts['trainable']:,}\n")
        f.write(f"Non-trainable parameters: {param_counts['non_trainable']:,}\n")
        
        # Add layer information
        f.write("\n" + "="*60 + "\n")
        f.write("LAYER INFORMATION\n")
        f.write("="*60 + "\n")
        
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf module
                params = sum(p.numel() for p in module.parameters())
                if params > 0:
                    f.write(f"{name}: {params:,} parameters\n")
    
    print(f"Model architecture saved to {save_path}")


def compare_models(model1: nn.Module, model2: nn.Module) -> Dict[str, Union[bool, str]]:
    """
    Compare two models for compatibility.
    
    Args:
        model1: First model
        model2: Second model
        
    Returns:
        Dictionary containing comparison results
    """
    comparison = {
        'compatible': True,
        'differences': [],
        'warnings': []
    }
    
    # Check parameter counts
    params1 = count_parameters(model1)
    params2 = count_parameters(model2)
    
    if params1['total'] != params2['total']:
        comparison['compatible'] = False
        comparison['differences'].append(f"Parameter count mismatch: {params1['total']} vs {params2['total']}")
    
    # Check model size
    size1 = get_model_size(model1)
    size2 = get_model_size(model2)
    
    if size1['total_size_mb'] != size2['total_size_mb']:
        comparison['warnings'].append(f"Model size mismatch: {size1['total_size_mb']:.2f} MB vs {size2['total_size_mb']:.2f} MB")
    
    # Check layer structure
    layers1 = set(name for name, _ in model1.named_modules())
    layers2 = set(name for name, _ in model2.named_modules())
    
    missing_in_2 = layers1 - layers2
    missing_in_1 = layers2 - layers1
    
    if missing_in_2:
        comparison['differences'].append(f"Layers missing in model2: {missing_in_2}")
    
    if missing_in_1:
        comparison['differences'].append(f"Layers missing in model1: {missing_in_1}")
    
    return comparison


def create_model_report(model: nn.Module, save_path: str) -> None:
    """
    Create a comprehensive model report.
    
    Args:
        model: PyTorch model
        save_path: Path to save the report
    """
    with open(save_path, 'w') as f:
        f.write("WAVEPOSE MODEL REPORT\n")
        f.write("=" * 60 + "\n\n")
        
        # Model architecture
        f.write("MODEL ARCHITECTURE\n")
        f.write("-" * 30 + "\n")
        f.write(str(model))
        f.write("\n\n")
        
        # Parameter information
        f.write("PARAMETER INFORMATION\n")
        f.write("-" * 30 + "\n")
        param_counts = count_parameters(model)
        f.write(f"Total parameters: {param_counts['total']:,}\n")
        f.write(f"Trainable parameters: {param_counts['trainable']:,}\n")
        f.write(f"Non-trainable parameters: {param_counts['non_trainable']:,}\n")
        f.write("\n")
        
        # Model size
        f.write("MODEL SIZE\n")
        f.write("-" * 30 + "\n")
        model_size = get_model_size(model)
        f.write(f"Total size: {model_size['total_size_mb']:.2f} MB\n")
        f.write(f"Trainable size: {model_size['trainable_size_mb']:.2f} MB\n")
        f.write(f"Precision: {model_size['precision']}\n")
        f.write("\n")
        
        # Layer breakdown
        f.write("LAYER BREAKDOWN\n")
        f.write("-" * 30 + "\n")
        total_params = 0
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf module
                params = sum(p.numel() for p in module.parameters())
                if params > 0:
                    f.write(f"{name}: {params:,} parameters\n")
                    total_params += params
        
        f.write(f"\nTotal layers with parameters: {total_params:,}\n")
    
    print(f"Model report saved to {save_path}")
