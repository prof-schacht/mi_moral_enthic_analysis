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
from .vector_utils import project_to_vocab

def plot_neuron_contributions(U: torch.Tensor, 
                            vector_ids: List[str],
                            S: torch.Tensor,
                            n_top: int = 15,
                            figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
    """
    Create heatmap visualization of neuron contributions to SVD components.
    
    Args:
        U: Left singular vectors [n_vectors, n_components]
        vector_ids: List of vector identifiers
        S: Singular values
        n_top: Number of top neurons to show per component
        figsize: Figure size (width, height)
        
    Returns:
        fig: matplotlib Figure object
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, gridspec_kw={'width_ratios': [3, 1]})
    
    # Prepare data for heatmap
    n_components = U.shape[1]
    contribution_data = []
    
    for comp_idx in range(n_components):
        scores = U[:, comp_idx].abs()
        top_indices = torch.argsort(scores, descending=True)[:n_top]
        
        for idx in top_indices:
            contribution_data.append({
                'Neuron': vector_ids[idx],
                'Component': f'SVD.UToxic[{comp_idx}]',
                'Contribution': U[idx, comp_idx].item()
            })
    
    df = pd.DataFrame(contribution_data)
    pivot_table = df.pivot(index='Neuron', columns='Component', values='Contribution')
    
    # Plot heatmap
    sns.heatmap(pivot_table, cmap='RdBu', center=0, ax=ax1)
    ax1.set_title('Neuron Contributions to SVD Components')
    
    # Plot explained variance
    variance_explained = (S.detach().cpu() ** 2) / (S.detach().cpu() ** 2).sum()
    ax2.bar(range(len(variance_explained)), variance_explained.detach().cpu())
    ax2.set_title('Explained Variance by Component')
    ax2.set_xlabel('Component')
    ax2.set_ylabel('Proportion of Variance Explained')
    
    plt.tight_layout()
    return fig

def plot_neuron_correlation_matrix(correlation_matrix: np.ndarray, 
                                 vector_ids: List[str],
                                 figsize: Tuple[int, int] = (12, 10)) -> None:
    """
    Create hierarchically clustered heatmap of neuron correlations.
    
    Args:
        correlation_matrix: Matrix of correlations between neurons
        vector_ids: List of vector identifiers
        figsize: Figure size (width, height)
    """
    plt.figure(figsize=figsize)
    
    # Perform hierarchical clustering
    linkage = hierarchy.linkage(correlation_matrix, method='ward')
    
    # Create clustered heatmap
    g = sns.clustermap(correlation_matrix,
                      xticklabels=vector_ids,
                      yticklabels=vector_ids,
                      cmap='RdBu_r',
                      center=0,
                      row_linkage=linkage,
                      col_linkage=linkage,
                      figsize=figsize)
    
    plt.setp(g.ax_heatmap.get_xticklabels(), rotation=45, ha='right')
    plt.title('Hierarchically Clustered Neuron Correlations')
    return g

def plot_neuron_graph(G: nx.Graph, figsize: Tuple[int, int] = (12, 8)) -> None:
    """
    Visualize neuron interaction graph.
    
    Args:
        G: NetworkX graph object
        figsize: Figure size (width, height)
    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Position nodes using force-directed layout
    pos = nx.spring_layout(G)
    
    # Draw nodes
    layers = nx.get_node_attributes(G, 'layer')
    unique_layers = sorted(set(layers.values()))
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_layers)))
    
    for layer, color in zip(unique_layers, colors):
        nodes = [n for n, l in layers.items() if l == layer]
        nx.draw_networkx_nodes(G, pos, 
                             nodelist=nodes,
                             node_color=[color],
                             node_size=500,
                             alpha=0.6,
                             ax=ax)
    
    # Draw edges
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    
    nx.draw_networkx_edges(G, pos,
                          edge_color=weights,
                          edge_cmap=plt.cm.RdBu_r,
                          edge_vmin=-1,
                          edge_vmax=1,
                          width=2,
                          ax=ax)
    
    # Add labels
    nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdBu_r, norm=plt.Normalize(vmin=-1, vmax=1))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label='Correlation Strength')
    
    ax.set_title('Neuron Interaction Graph')
    ax.axis('off')
    return plt

def create_interactive_neuron_graph(corr_matrix: np.ndarray, 
                                  neuron_ids: List[str], 
                                  toxic_matrix: torch.Tensor, 
                                  model: HookedTransformer,
                                  min_correlation: float = 0.2) -> go.Figure:
    """
    Create an interactive neuron correlation graph using Plotly.
    
    Args:
        corr_matrix: Correlation matrix between neurons
        neuron_ids: List of neuron identifiers
        toxic_matrix: Matrix of value vectors [n_vectors, d_model]
        model: The transformer model
        min_correlation: Minimum absolute correlation to show
    """
    # Create network graph
    G = nx.Graph()
    
    # Add nodes with layer information
    for i, nid in enumerate(neuron_ids):
        layer = int(nid.split('^')[1])
        neuron = nid.split('_')[1].split('^')[0]
        G.add_node(i, neuron_id=nid, layer=layer, neuron_num=neuron)
    
    # Add edges for correlations above threshold
    edge_weights = []
    for i in range(len(neuron_ids)):
        for j in range(i+1, len(neuron_ids)):
            if abs(corr_matrix[i, j]) >= min_correlation:
                G.add_edge(i, j, weight=corr_matrix[i, j])
                edge_weights.append(corr_matrix[i, j])
    
    # Use force-directed layout
    pos = nx.spring_layout(G, k=1/np.sqrt(len(G.nodes())), iterations=50)
    
    # Create edge traces
    edge_traces = []
    
    # Create a separate trace for each edge to handle different colors
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        weight = G.edges()[edge]['weight']
        
        # Map correlation to color
        color = f'rgb({int(255*(1-weight))}, {int(255*(1-abs(weight)))}, {int(255*(1+weight))})'
        
        edge_trace = go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            line=dict(width=2, color=color),
            hoverinfo='text',
            text=f'Correlation: {weight:.3f}',
            mode='lines',
            showlegend=False
        )
        edge_traces.append(edge_trace)
    
    # Create node traces by layer
    node_traces = []
    layers = sorted(set(nx.get_node_attributes(G, 'layer').values()))
    
    # Create color map for layers using a continuous colorscale
    n_layers = len(layers)
    colors = [f'hsl({h},70%,50%)' for h in np.linspace(0, 360, n_layers)]
    layer_colors = dict(zip(layers, colors))
    
    for layer in layers:
        node_x = []
        node_y = []
        node_text = []
        node_sizes = []
        
        for node in G.nodes():
            if G.nodes[node]['layer'] == layer:
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                
                neuron_id = G.nodes[node]['neuron_id']
                n_connections = len(list(G.neighbors(node)))
                
                # Get top tokens using existing project_to_vocab function
                vector_idx = neuron_ids.index(neuron_id)
                top_tokens = project_to_vocab(toxic_matrix[vector_idx], model, k=5)
                tokens_text = '<br>'.join([f"Token {i+1}: {token}" for i, token in enumerate(top_tokens)])
                
                node_text.append(
                    f"Layer: {layer}<br>"
                    f"Neuron: {G.nodes[node]['neuron_num']}<br>"
                    f"Connections: {n_connections}<br>"
                    f"<br>Top Tokens:<br>{tokens_text}"
                )
                
                node_sizes.append(10 + n_connections * 2)
        
        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers',
            hoverinfo='text',
            text=node_text,
            marker=dict(
                size=node_sizes,
                color=layer_colors[layer],
                line=dict(width=1, color='white')
            ),
            name=f'Layer {layer}'
        )
        node_traces.append(node_trace)
    
    # Create figure
    fig = go.Figure(
        data=edge_traces + node_traces,
        layout=go.Layout(
            title='Neuron Interaction Network',
            title_x=0.5,
            showlegend=True,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=1.05
            )
        )
    )
    
    return fig

def plot_layer_analysis(layer_stats: Dict, 
                       neuron_counts: Dict, 
                       figsize: Tuple[int, int] = (20, 8)) -> plt.Figure:
    """
    Create a side-by-side visualization of layer interactions and neuron counts.
    
    Args:
        layer_stats: Dictionary containing layer interaction statistics
        neuron_counts: Dictionary containing neuron counts per layer
        figsize: Figure size (width, height)
        
    Returns:
        matplotlib Figure object
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Extract unique layer numbers and create interaction matrix
    layers = sorted(set([int(k.split('_')[1]) for k in layer_stats.keys()]))
    n_layers = len(layers)
    interaction_matrix = np.zeros((n_layers, n_layers))
    
    for i, l1 in enumerate(layers):
        for j, l2 in enumerate(layers):
            key = f"layer_{l1}_to_{l2}"
            interaction_matrix[i, j] = layer_stats[key]
    
    # Plot heatmap with min-max scaling and no annotations
    sns.heatmap(interaction_matrix,
                xticklabels=layers,
                yticklabels=layers,
                cmap='RdBu_r',
                center=None,
                vmin=np.min(interaction_matrix),
                vmax=np.max(interaction_matrix),
                annot=False,
                ax=ax1)
    
    ax1.set_title('Layer-wise Interaction Strength')
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('Layer')
    
    # Plot neuron counts
    layers_for_bar = sorted(neuron_counts.keys())
    counts = [neuron_counts[l] for l in layers_for_bar]
    
    ax2.bar(layers_for_bar, counts, color='royalblue')
    ax2.set_title('Neurons per Layer')
    ax2.set_xlabel('Layer')
    ax2.set_ylabel('Number of Neurons')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_neuron_graph(correlation_matrix: np.ndarray, 
                       vector_ids: List[str],
                       threshold: float = 0.5) -> nx.Graph:
    """
    Create a graph representation of neuron interactions.
    
    Args:
        correlation_matrix: Matrix of correlations between neurons
        vector_ids: List of vector identifiers
        threshold: Minimum correlation strength to include
        
    Returns:
        G: NetworkX graph object
    """
    G = nx.Graph()
    
    # Add nodes
    for idx, vid in enumerate(vector_ids):
        layer = int(vid.split('^')[1])
        G.add_node(vid, layer=layer)
    
    # Add edges for strong correlations
    for i in range(len(vector_ids)):
        for j in range(i+1, len(vector_ids)):
            if abs(correlation_matrix[i, j]) > threshold:
                G.add_edge(vector_ids[i], 
                          vector_ids[j], 
                          weight=correlation_matrix[i, j])
    
    return G
