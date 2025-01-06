import os
import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import torch.nn.functional as F


class TransformerInterpreter:
    def __init__(self, model: nn.Module, feature_columns: List[str], device: torch.device, debug: bool = True):
        self.model = model
        self.feature_columns = feature_columns
        self.device = device
        self.debug = debug

    def _debug_gradient_flow(self, tensor: torch.Tensor, name: str):
        """Debug helper for gradient flow"""
        if not self.debug:
            return

        print(f"\nGradient flow for {name}:")
        print(f"Shape: {tensor.shape}")
        print(f"Device: {tensor.device}")
        print(f"Requires grad: {tensor.requires_grad}")
        if tensor.requires_grad:
            print(f"Grad fn: {tensor.grad_fn}")
        if hasattr(tensor, 'grad') and tensor.grad is not None:
            print(f"Gradient stats:")
            print(f"  Mean: {tensor.grad.abs().mean().item():.6f}")
            print(f"  Max: {tensor.grad.abs().max().item():.6f}")
            print(f"  Non-zero elements: {(tensor.grad != 0).sum().item()}")

    def compute_integrated_gradients(self, input_sequence: torch.Tensor, target_class: int = 1) -> torch.Tensor:
        """Compute integrated gradients with enhanced debugging"""
        self.model.train()

        # Move input to device and ensure proper type
        input_sequence = input_sequence.to(self.device).float()
        batch_size = input_sequence.size(0)

        # Create baseline
        baseline = torch.zeros_like(input_sequence)
        steps = 20  # Reduced for memory efficiency

        # Use smaller sub-batches
        sub_batch_size = 4
        n_sub_batches = (batch_size + sub_batch_size - 1) // sub_batch_size
        all_gradients = []

        print(f"\nComputing integrated gradients:")
        print(f"Input shape: {input_sequence.shape}")
        print(f"Using {steps} steps with {n_sub_batches} sub-batches")

        for batch_idx in range(n_sub_batches):
            start_idx = batch_idx * sub_batch_size
            end_idx = min((batch_idx + 1) * sub_batch_size, batch_size)

            sub_gradients = []
            for step in range(steps):
                try:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    # Create interpolated input
                    alpha = step / (steps - 1)
                    interpolated = baseline[start_idx:end_idx] + alpha * (
                            input_sequence[start_idx:end_idx] - baseline[start_idx:end_idx])
                    interpolated.requires_grad_(True)

                    if step == 0 and batch_idx == 0:
                        self._debug_gradient_flow(interpolated, "Interpolated input")

                    # Forward pass
                    outputs = self.model(interpolated)

                    if step == 0 and batch_idx == 0:
                        self._debug_gradient_flow(outputs, "Model outputs")

                    # Compute loss
                    current_batch_size = end_idx - start_idx
                    targets = torch.full((current_batch_size,), target_class,
                                         dtype=torch.long, device=self.device)

                    loss = F.cross_entropy(outputs, targets)

                    if step == 0 and batch_idx == 0:
                        print(f"\nInitial loss: {loss.item()}")
                        print(f"Loss requires grad: {loss.requires_grad}")

                    # Compute gradients
                    self.model.zero_grad(set_to_none=False)
                    loss.backward(retain_graph=True)

                    if interpolated.grad is None:
                        print(f"Warning: No gradient at step {step}, batch {batch_idx}")
                        print(f"Loss value: {loss.item()}")
                        print(f"Output shape: {outputs.shape}")
                        continue

                    if step == 0 and batch_idx == 0:
                        self._debug_gradient_flow(interpolated, "After backward")

                    grad = interpolated.grad.clone()
                    sub_gradients.append(grad)

                except Exception as e:
                    print(f"Error at step {step}, batch {batch_idx}: {str(e)}")
                    continue

            if sub_gradients:
                # Average gradients for this sub-batch
                avg_grad = torch.stack(sub_gradients).mean(dim=0)
                all_gradients.append(avg_grad)

                if batch_idx == 0:
                    print(f"\nSub-batch {batch_idx} gradient stats:")
                    print(f"Mean: {avg_grad.abs().mean().item():.6f}")
                    print(f"Max: {avg_grad.abs().max().item():.6f}")

            # Clean up
            del sub_gradients
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Process all gradients
        if all_gradients:
            try:
                # Combine gradients from all sub-batches
                gradients = torch.cat(all_gradients, dim=0)
                print(f"\nFinal gradient stats:")
                print(f"Mean: {gradients.abs().mean().item():.6f}")
                print(f"Max: {gradients.abs().max().item():.6f}")

                # Compute attributions
                attributions = (input_sequence - baseline) * gradients
                return attributions.abs().mean(dim=-1)

            except Exception as e:
                print(f"Error computing final attributions: {str(e)}")

        return torch.zeros_like(input_sequence[:, :, 0])

    def interpret_sequence(self, input_sequence: torch.Tensor,
                           target_class: int = 1) -> Dict[str, torch.Tensor]:
        """Get both attributions and attention weights"""
        attributions = self.compute_integrated_gradients(input_sequence, target_class)

        with torch.no_grad():
            attention_weights = self.model.get_attention_weights(input_sequence)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {
            'attributions': attributions.cpu().numpy(),
            'attention': attention_weights.cpu().numpy() if attention_weights is not None else None,
            'max_attribution': attributions.max().item(),
            'mean_attribution': attributions.mean().item()
        }