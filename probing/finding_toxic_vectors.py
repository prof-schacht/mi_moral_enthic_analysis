# %%
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

# Detoxify
from detoxify import Detoxify

# %%

def get_detoxify_scores(text: str) -> dict:
    """
    Get toxicity scores using Detoxify model
    
    Args:
        text: Text to analyze
        
    Returns:
        Dictionary containing scores for different attributes
        
    Library Page: https://pypi.org/project/detoxify/
        
    """
    model = Detoxify('original')
    scores = model.predict(text)
    return scores

# %%
def get_toxic_value_vectors(model: HookedTransformer, 
                            probe_vector: torch.Tensor,
                            n_vectors: int = 128) -> Tuple[torch.Tensor, List[str]]:
    vectors_info = []
    
    for layer_idx in range(model.cfg.n_layers):
        value_matrix = model.blocks[layer_idx].mlp.W_out  # [d_mlp, d_model]
        #print(f"Layer {layer_idx} value_matrix shape: {value_matrix.shape}")
        
        # Calculate cosine similarity directly with value_matrix
        # No need to transpose since dimensions are already correct
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

def project_to_vocab(vector: torch.Tensor, model: HookedTransformer, k: int = 6) -> List[str]:
    """Project vector to vocabulary space and return top tokens."""
    #print(f"Vector shape in project_to_vocab: {vector.shape}")
    #print(f"Unembedding matrix shape: {model.unembed.W_U.shape}")
    logits = model.unembed.W_U.T @ vector  # Adjusted multiplication
    top_tokens = torch.topk(logits, k=k).indices
    return [model.tokenizer.decode(idx.item()).strip() for idx in top_tokens]


def create_formatted_table(toxic_vectors_info: List[dict], 
                         svd_vectors: torch.Tensor,
                         model: HookedTransformer) -> str:
    """Create formatted table like in the paper."""
    table_data = []
    
    # Add MLP vectors
    for i, info in enumerate(toxic_vectors_info):
        #print(f"Vector {i} shape: {info['vector'].shape}")
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

def print_warning():
    """Print warning message as in paper."""
    warning = """
WARNING: THESE EXAMPLES ARE HIGHLY OFFENSIVE.
We note that SVD.U_Toxic[2] has a particularly gendered nature.
This arises from the dataset and language model we use.
    """
    print(warning)

# %%


### Analyzing of SVD Vectors:
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
            
            
# %% Interconnection analysis
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

def plot_neuron_correlation_matrix(correlation_matrix: np.ndarray, 
                                 vector_ids: List[str],
                                 figsize: Tuple[int, int] = (12, 10)) -> None:
    """
    Create hierarchically clustered heatmap of neuron correlations.
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

def plot_neuron_graph(G: nx.Graph, figsize: Tuple[int, int] = (12, 8)) -> None:
    """
    Visualize neuron interaction graph.
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

def create_interactive_neuron_graph(corr_matrix: np.ndarray, neuron_ids: List[str], 
                                  toxic_matrix: torch.Tensor, model: HookedTransformer,
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


def analyze_layer_patterns(correlation_matrix: np.ndarray,
                         vector_ids: List[str]) -> Dict:
    """
    Analyze patterns of interactions between layers.
    
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

def plot_layer_analysis(layer_stats: Dict, neuron_counts: Dict, figsize: Tuple[int, int] = (20, 8)) -> plt.Figure:
    """
    Create a side-by-side visualization of layer interactions and neuron counts.
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
                center=None,  # Remove center to use min-max scaling
                vmin=np.min(interaction_matrix),
                vmax=np.max(interaction_matrix),
                annot=False,  # Remove numerical annotations
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

def get_neuron_counts(vector_ids: List[str]) -> Dict[int, int]:
    """
    Get the number of toxic neurons found per layer from the vector IDs.
    """
    neuron_counts = {}
    # Extract layer numbers from vector IDs (format: "MLP.v_X^Y" where Y is layer number)
    for vid in vector_ids:
        layer = int(vid.split('^')[1])
        neuron_counts[layer] = neuron_counts.get(layer, 0) + 1
    return neuron_counts

# %%


# Intervention 
def calculate_perplexity(model: HookedTransformer, 
                        dataset_name: str = "Salesforce/wikitext",
                        name: str = "wikitext-2-v1", 
                        split: str = "test",
                        max_samples: int = 100,
                        hook_fn: Optional[Callable] = None,
                        hook_point: Optional[str] = None) -> float:
    """
    Calculate model perplexity on a given dataset. We do this to ensure that the model has the same perplexity
    on the dataset before and after intervention. And we can proof that the intervention does not destroy the 
    performance of the model on the dataset.
    """
    # Load dataset
    dataset = load_dataset(dataset_name, name=name, split=split)
    
    # Setup
    total_loss = 0
    total_tokens = 0
    
    # Add hook if provided
    if hook_fn and hook_point:
        model.add_perma_hook(hook_point, hook_fn)
    
    model.eval()
    with torch.no_grad():
        for i in range(min(max_samples, len(dataset))):
            # Ensure text is not empty
            text = dataset[i]['text']
            if not text.strip():
                continue
                
            # Configure tokenizer
            model.tokenizer.padding_side = 'right'
            model.tokenizer.pad_token = model.tokenizer.eos_token
            
            # Tokenize text
            tokens = model.tokenizer(
                text,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=512,
                return_attention_mask=True
            )
            
            # Skip empty sequences
            if tokens['input_ids'].size(1) == 0:
                continue
                
            # Move to device
            input_ids = tokens['input_ids'].to(model.cfg.device)
            attention_mask = tokens['attention_mask'].to(model.cfg.device)
            
            # Forward pass
            logits = model(input_ids)  # Let the model handle attention mask internally
            
            # Calculate loss
            loss = cross_entropy(
                logits[0, :-1, :].reshape(-1, logits.size(-1)),
                input_ids[0, 1:].reshape(-1),
                reduction='sum'
            )
            
            # Update totals
            total_loss += loss.item()
            total_tokens += (input_ids.size(1) - 1)  # -1 because we don't predict first token
    
    # Remove hook if it was added
    if hook_fn and hook_point:
        model.reset_hooks()
    
    # Calculate perplexity
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    return perplexity

def last_layer_intervention(model: HookedTransformer,
                          prompt: str,
                          toxic_vector: torch.Tensor,
                          alpha: float = 1.0,
                          max_tokens: int = 50,
                          calculate_metrics: bool = True) -> Dict[str, Union[str, float]]:
    """
    Demonstrate last layer intervention by subtracting toxic vector.
    Returns both original and intervened outputs along with perplexity scores.
    
    Args:
        model: HookedTransformer model
        prompt: Input prompt
        toxic_vector: Vector to subtract
        alpha: Scaling factor for intervention
        max_tokens: Maximum tokens to generate
        calculate_metrics: Whether to calculate perplexity scores
        
    Returns:
        Dictionary containing outputs and metrics
    """
    # Calculate which layer is the last layer
    last_layer = model.cfg.n_layers - 1
    hook_point = f"blocks.{last_layer}.hook_resid_post"
    
    # First generate without hook
    original_output = model.generate(prompt, max_new_tokens=max_tokens, temperature=0.7)
    
    def intervention_hook(activation, hook):
        expanded_vector = toxic_vector.unsqueeze(0).unsqueeze(0)
        return activation - alpha * expanded_vector
    
    # Calculate baseline perplexity if requested
    results = {
        'baseline_output': original_output,
    }
    
    if calculate_metrics:
        baseline_perplexity = calculate_perplexity(model)
        results['baseline_perplexity'] = baseline_perplexity
        results['baseline_toxic_metrics'] = get_detoxify_scores(original_output)
    
    # Add the hook and generate
    model.add_perma_hook(hook_point, intervention_hook)
    
    # Generate with hook and calculate perplexity if requested
    hooked_output = model.generate(prompt, max_new_tokens=max_tokens, temperature=0.7)
    results['hooked_output'] = hooked_output
    
    if calculate_metrics:
        hooked_perplexity = calculate_perplexity(
            model,
            hook_fn=intervention_hook,
            hook_point=hook_point
        )
        results['hooked_perplexity'] = hooked_perplexity
        results['hooked_toxic_metrics'] = get_detoxify_scores(hooked_output)
    # Clean up - remove all hooks
    model.reset_hooks()
    
    return results

# %%
def main():
    # %%
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # Load model
    model = HookedTransformer.from_pretrained(
        "gpt2-medium",
        center_writing_weights=False,
        center_unembed=False,
        fold_ln=False,
        device=device
    )
    
    # %%
    
    # Load probe and move to correct device
    probe_state = torch.load('toxicity_probe_single_vector.pt', weights_only=True)
    #probe_state = torch.load('toxicity_probe_mt_vectors.pt', weights_only=True)
    probe_vector = probe_state['classifier.weight'][1].to(device)  # toxic class vector
    
    # %%
    
    # Get toxic vectors
    toxic_matrix, vector_ids, top_15_info = get_toxic_value_vectors(model, probe_vector, n_vectors=128)
    
    # %%
    
    # Compute SVD directly on this matrix
    U, S, V = torch.svd(toxic_matrix)
    
    # %%

    # Take first 3 principal components (right singular vectors)
    svd_vectors = V[:, :3]  # [d_model, 3]

    # %%
    # Project back to model space
    svd_vectors = toxic_matrix.T @ (U[:, :3] / S[:3].unsqueeze(0))  # [1024, 3]
    
    # %%
    # Print warning
    print_warning()
    
    # Create and print table
    print("\nTable 1. Top toxic vectors projected onto the vocabulary space.\n")
    table = create_formatted_table(top_15_info, svd_vectors, model)
    print(table)


    # Analyzing the SVD vectors
    # Analyze contributions
    U, S, V = analyze_neuron_contributions(toxic_matrix, vector_ids)

    # Get and print top neurons per component
    components_info = get_top_neurons_per_component(U, vector_ids)
    print_component_analysis(components_info, model, toxic_matrix, vector_ids)

    # Create visualization
    fig = plot_neuron_contributions(U, vector_ids, S)
    fig.savefig("./plots/neuron_contributions.png")
    
    
    # Analyze interactions
    corr_matrix, ids = analyze_neuron_interactions(model, toxic_matrix, vector_ids)

    # Create visualizations
    fig = plot_neuron_correlation_matrix(corr_matrix, ids)
    fig.savefig("./plots/neuron_correlation_matrix.png")

    # Create and visualize interaction graph
    G = create_neuron_graph(corr_matrix, ids)
    fig = plot_neuron_graph(G)
    fig.savefig("./plots/neuron_interaction_graph.png")

    # Analyze layer patterns
    layer_stats = analyze_layer_patterns(corr_matrix, ids)
    print("Layer interaction patterns:")
    for k, v in layer_stats.items():
        print(f"{k}: {v:.3f}")
        
    # Get neuron counts and plot layer analysis
    neuron_counts = get_neuron_counts(vector_ids)
    fig = plot_layer_analysis(layer_stats, neuron_counts)
    fig.savefig("./plots/layer_analysis.png", dpi=300, bbox_inches='tight')

    # Create and save interactive plot with existing data
    fig = create_interactive_neuron_graph(corr_matrix, ids, toxic_matrix, model)
    fig.write_html("./plots/neuron_interactions.html")


    # Invervene toxic vector to last layer
    
    
    
    
# %%

if __name__ == "__main__":
    main()