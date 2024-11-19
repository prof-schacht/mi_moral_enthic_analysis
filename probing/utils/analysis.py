import torch
import numpy as np
from transformer_lens import HookedTransformer
from typing import List, Tuple, Dict, Optional, Callable, Union
from tabulate import tabulate
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.cluster import hierarchy
from scipy.stats import pearsonr
from scipy.spatial.distance import squareform
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from datasets import load_dataset
from torch.nn.functional import cross_entropy
from utils.vector_utils import project_to_vocab

from transformer_lens import HookedTransformer

def analyze_neuron_contributions(toxic_matrix: torch.Tensor,
                               vector_ids: List[str],
                               n_components: int = 3) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Analyze how each original neuron contributes to SVD components.
    
    Args:
        toxic_matrix: Matrix of value vectors [n_vectors, d_model]
        vector_ids: List of vector identifiers
        n_components: Number of SVD components to analyze
        
    Returns:
        U: Left singular vectors showing neuron contributions
        S: Singular values
        V: Right singular vectors (the SVD components)
    """
    U, S, V = torch.svd(toxic_matrix)
    return U[:, :n_components], S[:n_components], V[:, :n_components]

def get_top_neurons_per_component(U: torch.Tensor,
                                vector_ids: List[str],
                                n_top: int = 10) -> List[Dict]:
    """
    Get top contributing neurons for each SVD component.
    
    Args:
        U: Left singular vectors [n_vectors, n_components]
        vector_ids: List of vector identifiers
        n_top: Number of top neurons to return per component
        
    Returns:
        List of dictionaries containing top neurons for each component
    """
    components_info = []
    
    for component_idx in range(U.shape[1]):
        # Get contribution scores for this component
        scores = U[:, component_idx].abs()
        
        # Get indices of top contributing neurons
        top_indices = torch.argsort(scores, descending=True)[:n_top]
        
        # Create component info
        component_info = {
            'component': f'SVD.UToxic[{component_idx}]',
            'neurons': [vector_ids[idx] for idx in top_indices],
            'contributions': scores[top_indices].tolist()
        }
        components_info.append(component_info)
    
    return components_info

def analyze_neuron_interactions(model, value_vectors: torch.Tensor,
                              vector_ids: List[str],
                              n_samples: int = 1000) -> Tuple[np.ndarray, List[str]]:
    """
    Analyze interactions between neurons by looking at their activation patterns.
    
    Args:
        model: The transformer model
        value_vectors: Matrix of value vectors [n_vectors, d_model]
        vector_ids: List of vector identifiers
        n_samples: Number of samples to use for correlation analysis
        
    Returns:
        correlation_matrix: Matrix of neuron correlations
        filtered_ids: List of vector IDs that were analyzed
    """
    # Generate random input to get activations
    activations = []
    for i in range(n_samples):
        # Random input
        x = torch.randn(1, model.cfg.d_model).to(value_vectors.device)
        
        # Get activations for each vector
        acts = []
        for vec in value_vectors:
            activation = torch.nn.functional.cosine_similarity(
                x, vec.unsqueeze(0), dim=1
            )
            acts.append(activation.item())
        activations.append(acts)
    
    # Convert to numpy array
    activations = np.array(activations)
    
    # Calculate correlation matrix
    correlation_matrix = np.zeros((len(vector_ids), len(vector_ids)))
    for i in range(len(vector_ids)):
        for j in range(len(vector_ids)):
            corr, _ = pearsonr(activations[:, i], activations[:, j])
            correlation_matrix[i, j] = corr
            
    return correlation_matrix, vector_ids

def analyze_layer_patterns(correlation_matrix: np.ndarray,
                         vector_ids: List[str]) -> Dict:
    """
    Analyze patterns of interactions between layers.
    
    Args:
        correlation_matrix: Matrix of correlations between neurons
        vector_ids: List of vector identifiers
        
    Returns:
        Dictionary containing layer interaction statistics
    """
    layers = [int(vid.split('^')[1]) for vid in vector_ids]
    unique_layers = sorted(set(layers))
    
    layer_stats = {}
    
    # Calculate average correlation between and within layers
    for l1 in unique_layers:
        for l2 in unique_layers:
            idx1 = [i for i, l in enumerate(layers) if l == l1]
            idx2 = [i for i, l in enumerate(layers) if l == l2]
            
            corrs = correlation_matrix[np.ix_(idx1, idx2)]
            avg_corr = np.mean(corrs)
            
            key = f"layer_{l1}_to_{l2}"
            layer_stats[key] = avg_corr
    
    return layer_stats

def get_neuron_counts(vector_ids: List[str]) -> Dict[int, int]:
    """
    Get the number of toxic neurons found per layer from the vector IDs.
    
    Args:
        vector_ids: List of vector identifiers
        
    Returns:
        Dictionary mapping layer numbers to neuron counts
    """
    neuron_counts = {}
    # Extract layer numbers from vector IDs (format: "MLP.v_X^Y" where Y is layer number)
    for vid in vector_ids:
        layer = int(vid.split('^')[1])
        neuron_counts[layer] = neuron_counts.get(layer, 0) + 1
    return neuron_counts


def print_component_analysis(components_info: List[Dict], model: HookedTransformer, toxic_matrix: torch.Tensor, vector_ids: List[str]) -> None:
    """
    Print formatted analysis of top neurons per component with their tokens.
    
    Args:
        components_info: List of dictionaries containing component information
        model: HookedTransformer model for token projection
        toxic_matrix: Matrix of value vectors [n_vectors, d_model]
        vector_ids: List of vector identifiers
    """
    for info in components_info:
        print(f"\n{info['component']} Top Contributing Neurons:")
        print("-" * 80)
        print(f"{'Neuron':<15} | {'Contribution':^12} | {'Top Tokens'}")
        print("-" * 80)
        
        for neuron, contribution in zip(info['neurons'], info['contributions']):
            # Find the vector corresponding to this neuron
            neuron_idx = vector_ids.index(neuron)
            vector = toxic_matrix[neuron_idx]
            
            # Get top tokens for this vector
            tokens = project_to_vocab(vector, model, k=4)
            tokens_str = ", ".join(tokens)
            
            print(f"{neuron:<15} | {contribution:>10.4f} | {tokens_str}")