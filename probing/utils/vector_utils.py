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

def get_toxic_value_vectors(model: HookedTransformer, 
                          probe_vector: torch.Tensor,
                          n_vectors: int = 128) -> Tuple[torch.Tensor, List[str]]:
    vectors_info = []
    
    for layer_idx in range(model.cfg.n_layers):
        value_matrix = model.blocks[layer_idx].mlp.W_out  # [d_mlp, d_model]
        
        # Calculate cosine similarity directly with value_matrix
        sims = torch.nn.functional.cosine_similarity(
            probe_vector.unsqueeze(0),  # [1, d_model]
            value_matrix,               # [d_mlp, d_model]
            dim=1
        )
        
        for vector_idx in range(value_matrix.shape[0]):
            # Create one-hot vector
            one_hot = torch.nn.functional.one_hot(
                torch.tensor(vector_idx), 
                num_classes=value_matrix.shape[0]
            ).to(value_matrix.device).to(value_matrix.dtype)
            
            # Project to d_model space - transpose value_matrix for correct multiplication
            projected_vector = value_matrix.T @ one_hot  # [d_model]
            
            vectors_info.append({
                'vector': projected_vector,
                'similarity': sims[vector_idx].item(),
                'identifier': f"MLP.v_{vector_idx}^{layer_idx}"
            })
    
    # Sort by similarity
    vectors_info.sort(key=lambda x: x['similarity'], reverse=True)
    
    # Get top N
    top_vectors = vectors_info[:n_vectors]
    matrix = torch.stack([info['vector'] for info in top_vectors])  # Will be [n_vectors, d_model]
    identifiers = [info['identifier'] for info in top_vectors]
    
    return matrix, identifiers, vectors_info[:15]

def create_formatted_table(toxic_vectors_info: List[dict], 
                         svd_vectors: torch.Tensor,
                         model: HookedTransformer) -> str:
    """Create formatted table like in the paper."""
    table_data = []
    
    # Add MLP vectors
    for i, info in enumerate(toxic_vectors_info):
        tokens = project_to_vocab(info['vector'], model, k=6)
        tokens_str = ", ".join(tokens)
        table_data.append([info['identifier'], tokens_str])
    
    # Add SVD vectors - now svd_vectors is already in model dimension
    for i in range(3):  # Top 3 SVD vectors as in paper
        tokens = project_to_vocab(svd_vectors[:, i], model, k=6)
        tokens_str = ", ".join(tokens)
        table_data.append([f"SVD.U_Toxic[{i}]", tokens_str])
    
    # Create table with headers
    headers = ["VECTOR", "TOP TOKENS"]
    return tabulate(table_data, headers=headers, tablefmt="pipe")

def project_to_vocab(vector: torch.Tensor, model: HookedTransformer, k: int = 6) -> List[str]:
    """Project vector to vocabulary space and return top tokens."""
    logits = model.unembed.W_U.T @ vector  # Adjusted multiplication
    top_tokens = torch.topk(logits, k=k).indices
    return [model.tokenizer.decode(idx.item()).strip() for idx in top_tokens]