import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from transformer_lens import HookedTransformer
from typing import List, Tuple, Dict

def get_layer_value_vectors(model: HookedTransformer) -> List[torch.Tensor]:
    """Extract value vectors from each layer of the model."""
    value_vectors_by_layer = []
    for layer_idx in range(model.cfg.n_layers):
        # Get value matrix [d_mlp, d_model]
        value_matrix = model.blocks[layer_idx].mlp.W_out
        value_vectors_by_layer.append(value_matrix)
    return value_vectors_by_layer

def plot_cross_layer_analysis(model: HookedTransformer, 
                            probe_vector: torch.Tensor,
                            n_clusters: int = 5,
                            batch_size: int = 1000):
    """
    Create comprehensive visualization of value vectors across layers.
    
    Args:
        model: HookedTransformer model
        probe_vector: Toxicity probe vector [d_model]
        n_clusters: Number of clusters for hierarchical clustering
        batch_size: Batch size for cosine similarity computation
    """
    value_vectors_by_layer = get_layer_value_vectors(model)
    n_layers = len(value_vectors_by_layer)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 15))
    gs = plt.GridSpec(2, 2, figure=fig)
    
    # 1. Similarity Heatmap between layers
    ax1 = fig.add_subplot(gs[0, 0])
    layer_similarities = torch.zeros((n_layers, n_layers))
    
    for i in range(n_layers):
        for j in range(n_layers):
            # Process in batches
            vectors_i = value_vectors_by_layer[i]
            vectors_j = value_vectors_by_layer[j]
            n_vectors = vectors_i.shape[0]
            batch_sims = []
            
            for start_idx in range(0, n_vectors, batch_size):
                end_idx = min(start_idx + batch_size, n_vectors)
                batch_i = vectors_i[start_idx:end_idx].unsqueeze(1)
                
                # Process second dimension in sub-batches
                sub_batch_sims = []
                for sub_start_idx in range(0, n_vectors, batch_size):
                    sub_end_idx = min(sub_start_idx + batch_size, n_vectors)
                    batch_j = vectors_j[sub_start_idx:sub_end_idx].unsqueeze(0)
                    
                    sims = torch.nn.functional.cosine_similarity(
                        batch_i,
                        batch_j,
                        dim=2
                    )
                    sub_batch_sims.append(sims.mean().item())
                
                batch_sims.extend(sub_batch_sims)
            
            layer_similarities[i, j] = torch.tensor(batch_sims).mean()
    
    sns.heatmap(layer_similarities.cpu().numpy(),
                ax=ax1,
                cmap='RdYlBu_r',
                xticklabels=[f'Layer {i}' for i in range(n_layers)],
                yticklabels=[f'Layer {i}' for i in range(n_layers)])
    ax1.set_title('Cross-Layer Similarity Heatmap')
    
    # 2. Probe Vector Similarities
    ax2 = fig.add_subplot(gs[0, 1])
    probe_sims = []
    for layer_vectors in value_vectors_by_layer:
        sims = torch.nn.functional.cosine_similarity(
            probe_vector.unsqueeze(0),
            layer_vectors,
            dim=1
        )
        probe_sims.append(sims.detach().cpu().numpy())
    
    # Plot violin plot of similarities
    plt.violinplot(probe_sims, positions=range(n_layers))
    ax2.set_xticks(range(n_layers))
    ax2.set_xticklabels([f'Layer {i}' for i in range(n_layers)])
    ax2.set_title('Distribution of Similarities to Toxicity Probe')
    ax2.set_ylabel('Cosine Similarity')
    
    # 3. Hierarchical Clustering Visualization
    ax3 = fig.add_subplot(gs[1, 0])
    # Stack all vectors and convert to numpy for clustering
    all_vectors = torch.cat([v.cpu() for v in value_vectors_by_layer], dim=0).detach().numpy()
    linkage_matrix = linkage(all_vectors, method='ward')
    dendrogram(linkage_matrix, ax=ax3, truncate_mode='lastp', p=n_clusters)
    ax3.set_title('Hierarchical Clustering of Value Vectors')
    
    # 4. Vector Magnitudes Distribution
    ax4 = fig.add_subplot(gs[1, 1])
    activation_strengths = []
    for vectors in value_vectors_by_layer:
        strengths = torch.norm(vectors, dim=1).detach().cpu().numpy()
        activation_strengths.append(strengths)
    
    plt.boxplot(activation_strengths, labels=[f'Layer {i}' for i in range(n_layers)])
    ax4.set_title('Distribution of Vector Magnitudes Across Layers')
    ax4.set_ylabel('L2 Norm')
    
    plt.tight_layout()
    return fig

def plot_vector_evolution(model: HookedTransformer, 
                         vector_indices: List[int],
                         probe_vector: torch.Tensor):
    """
    Plot how specific vectors evolve across layers.
    
    Args:
        model: HookedTransformer model
        vector_indices: List of vector indices to track
        probe_vector: Toxicity probe vector
    """
    plt.figure(figsize=(12, 6))
    n_layers = model.cfg.n_layers
    
    for idx in vector_indices:
        values = []
        for layer_idx in range(n_layers):
            value_matrix = model.blocks[layer_idx].mlp.W_out
            vector = value_matrix[idx]
            # Compute similarity with probe
            sim = torch.nn.functional.cosine_similarity(
                probe_vector.unsqueeze(0),
                vector.unsqueeze(0),
                dim=1
            )
            values.append(sim.item())
        
        plt.plot(values, label=f'Vector {idx}', marker='o')
    
    plt.xlabel('Layer')
    plt.ylabel('Similarity with Toxicity Probe')
    plt.title('Evolution of Vector-Probe Similarity Across Layers')
    plt.legend()
    plt.grid(True)
    return plt.gcf()

def plot_concept_correlations(model: HookedTransformer, 
                            concept_probes: Dict[str, torch.Tensor]):
    """
    Plot correlations between different concept probes across layers.
    
    Args:
        model: HookedTransformer model
        concept_probes: Dictionary mapping concept names to probe vectors
    """
    n_layers = model.cfg.n_layers
    n_concepts = len(concept_probes)
    correlations = torch.zeros((n_concepts, n_layers))
    
    for i, (concept_name, probe) in enumerate(concept_probes.items()):
        for layer_idx in range(n_layers):
            value_matrix = model.blocks[layer_idx].mlp.W_out
            sims = torch.nn.functional.cosine_similarity(
                probe.unsqueeze(0),
                value_matrix,
                dim=1
            )
            correlations[i, layer_idx] = sims.mean()
    
    plt.figure(figsize=(12, 6))
    sns.heatmap(correlations.cpu().detach().numpy(),
                xticklabels=[f'Layer {i}' for i in range(n_layers)],
                yticklabels=list(concept_probes.keys()),
                cmap='RdYlBu_r')
    plt.title('Concept Correlations Across Layers')
    plt.xlabel('Layer')
    plt.ylabel('Concept')
    return plt.gcf()

# Example usage:
def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = HookedTransformer.from_pretrained(
        "gpt2-medium",
        center_writing_weights=False,
        center_unembed=False,
        fold_ln=False,
        device=device
    )
    
    # Load probe
    probe_state = torch.load('toxicity_probe_single_vector.pt', weights_only=True)
    probe_vector = probe_state['classifier.weight'][1].to(device)
    
    # Create main analysis plot with smaller batch size
    fig = plot_cross_layer_analysis(model, probe_vector, batch_size=500)  # Adjust batch_size as needed
    plt.savefig('cross_layer_analysis.png')
    
    # Plot evolution of top toxic vectors
    # Get indices of top toxic vectors (you can modify this based on your needs)
    top_indices = [0, 1, 2]  # Example indices
    fig_evolution = plot_vector_evolution(model, top_indices, probe_vector)
    plt.savefig('vector_evolution.png')
    
    # Plot concept correlations (if you have multiple concept probes)
    concept_probes = {
        'Toxic': probe_vector,
        # Add more concept probes if available
    }
    fig_correlations = plot_concept_correlations(model, concept_probes)
    plt.savefig('concept_correlations.png')

if __name__ == "__main__":
    main()