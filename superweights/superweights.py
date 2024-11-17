import torch
from transformers import AutoTokenizer
from transformer_lens import HookedTransformer
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
import seaborn as sns

class SuperWeightDetector:
    def __init__(self, model_name: str = "gpt2-medium"):
        """Initialize the detector with a specific model."""
        self.model = HookedTransformer.from_pretrained(
            model_name,
            center_writing_weights=False,
            center_unembed=False,
            fold_ln=False,

    )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
    def get_mlp_activations(self, 
                           prompt: str, 
                           layer: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the input and output activations of the MLP down projection for a specific layer.
        """
        activations = {}
        
        def hook_mlp_in(act, hook):
            activations['mlp_in'] = act.detach()
            
        def hook_mlp_out(act, hook):
            activations['mlp_out'] = act.detach()
        
        # Register hooks using the correct hook names
        hooks = [
            (f'blocks.{layer}.mlp.hook_pre', hook_mlp_in),
            (f'blocks.{layer}.mlp.hook_post', hook_mlp_out)
        ]
        
        # Run forward pass with hooks
        tokens = self.tokenizer.encode(prompt, return_tensors="pt")
        with torch.no_grad():
            _ = self.model.run_with_hooks(
                tokens,
                fwd_hooks=hooks
            )
            
        return activations['mlp_in'], activations['mlp_out']

    def find_activation_spikes(self, prompt: str, threshold: float = 3.0):
        """
        Find potential super weights by analyzing activation spikes across layers.
        """
        spikes = []
        
        for layer in range(self.model.cfg.n_layers):
            mlp_in, mlp_out = self.get_mlp_activations(prompt, layer)
            
            # Get the actual dimensions from the weight matrix
            W_out = self.model.blocks[layer].mlp.W_out
            d_model, d_mlp = W_out.shape  # Should be [1024, 4096] for gpt2-medium
            
            # Reshape activations for easier analysis
            mlp_in_reshaped = mlp_in.reshape(-1, mlp_in.shape[-1])
            mlp_out_reshaped = mlp_out.reshape(-1, mlp_out.shape[-1])
            
            # Find maximum values and their positions
            max_in_vals, max_in_pos = torch.max(mlp_in_reshaped, dim=1)
            max_out_vals, max_out_pos = torch.max(mlp_out_reshaped, dim=1)
            
            # Calculate statistics for thresholding
            mean_in, std_in = torch.mean(max_in_vals), torch.std(max_in_vals)
            mean_out, std_out = torch.mean(max_out_vals), torch.std(max_out_vals)
            
            # Find high activation positions
            high_in_mask = max_in_vals > (mean_in + threshold * std_in)
            high_out_mask = max_out_vals > (mean_out + threshold * std_out)
            
            # Get indices where both input and output show high activations
            for idx in torch.nonzero(high_in_mask & high_out_mask, as_tuple=True)[0]:
                # Ensure coordinates are within bounds of the weight matrix
                row = int(max_out_pos[idx].item()) % d_model  # Output dimension (1024)
                col = int(max_in_pos[idx].item()) % d_mlp     # Input dimension (4096)
                activation_value = float(max_out_vals[idx].item())
                
                # Double-check that indices are within bounds
                if row < d_model and col < d_mlp:
                    spikes.append((
                        layer,          # Layer number
                        (row, col),     # 2D coordinates (row, col)
                        activation_value # Activation value
                    ))
        
        return sorted(spikes, key=lambda x: abs(x[2]), reverse=True)

    def visualize_layer_activations(self, prompt: str, layer: int):
        """
        Visualize the activation patterns for a specific layer.
        """
        mlp_in, mlp_out = self.get_mlp_activations(prompt, layer)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot input activations
        sns.heatmap(mlp_in[0].cpu().numpy(), ax=ax1, cmap='viridis')
        ax1.set_title(f'Layer {layer} MLP Down Projection Input')
        
        # Plot output activations
        sns.heatmap(mlp_out[0].cpu().numpy(), ax=ax2, cmap='viridis')
        ax2.set_title(f'Layer {layer} MLP Down Projection Output')
        
        plt.tight_layout()
        return fig

    def verify_super_weight(self, prompt: str, layer: int, coordinates: Tuple[int, int]):
        """
        Verify potential super weight using 2D coordinates.
        """
        with torch.no_grad():
            tokens = self.tokenizer.encode(prompt, return_tensors="pt")
            original_output = self.model(tokens)
            
            # Access weight using 2D coordinates
            row, col = coordinates
            original_weight = self.model.blocks[layer].mlp.W_out[row, col].item()
            self.model.blocks[layer].mlp.W_out[row, col] = 0
            
            modified_output = self.model(tokens)
            self.model.blocks[layer].mlp.W_out[row, col] = original_weight
            
            output_diff = torch.norm(original_output - modified_output)
            return float(output_diff.item())

    def plot_max_activations_across_layers(self, prompt: str):
        """
        Plot maximum activation values for MLP down projection across all layers.
        """
        n_layers = self.model.cfg.n_layers
        max_in_values = []
        max_out_values = []
        
        # Collect maximum activations for each layer
        for layer in range(n_layers):
            mlp_in, mlp_out = self.get_mlp_activations(prompt, layer)
            
            # Get maximum values for each layer
            max_in = torch.max(mlp_in).item()
            max_out = torch.max(mlp_out).item()
            
            max_in_values.append(max_in)
            max_out_values.append(max_out)
        
        # Create the visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot input activations
        ax1.plot(range(n_layers), max_in_values, 'bo-')
        ax1.set_title(f'{self.model.cfg.model_name} Max down_proj Input')
        ax1.set_xlabel('Layer Number')
        ax1.set_ylabel('Max Activation Value')
        ax1.grid(True)
        
        # Plot output activations
        ax2.plot(range(n_layers), max_out_values, 'bo-')
        ax2.set_title(f'{self.model.cfg.model_name} Max down_proj Output')
        ax2.set_xlabel('Layer Number')
        ax2.set_ylabel('Max Activation Value')
        ax2.grid(True)
        
        plt.tight_layout()
        return fig

def main():
    detector = SuperWeightDetector("meta-llama/Llama-2-7b-hf")
    prompt = "My favorite condiment is"
    
    # Create and save the activation plot
    fig = detector.plot_max_activations_across_layers(prompt)
    plt.savefig('max_activations_across_layers.png')
    plt.close()
    
    spikes = detector.find_activation_spikes(prompt, threshold=1.0)
    print("\nPotential super weights found:")
    for layer, coords, value in spikes[:5]:
        print(f"Layer: {layer}, Coordinates: [{coords[0]}, {coords[1]}], Activation: {value:.2f}")
        
        # Verify impact
        impact = detector.verify_super_weight(prompt, layer, coords)
        print(f"Impact when zeroed: {impact:.2f}\n")
        
        # Optional: Print weight matrix dimensions for this layer
        weight_shape = detector.model.blocks[layer].mlp.W_out.shape
        print(f"Weight matrix dimensions at layer {layer}: {weight_shape}\n")
    
    # Visualize the layer with the strongest spike
    if spikes:
        layer_to_viz = spikes[0][0]
        fig = detector.visualize_layer_activations(prompt, layer_to_viz)
        plt.savefig('super_weights_layer_activations.png')


if __name__ == "__main__":
    main()