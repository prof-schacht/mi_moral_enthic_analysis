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

from transformer_lens import HookedTransformer

from utils.detoxify_utils import get_detoxify_scores, print_warning
from utils.vector_utils import (
    get_toxic_value_vectors,
    create_formatted_table,
    project_to_vocab
)
from utils.analysis import (
    analyze_neuron_contributions,
    get_top_neurons_per_component,
    analyze_neuron_interactions,
    analyze_layer_patterns,
    get_neuron_counts,
    print_component_analysis
)
from utils.visualization import (
    plot_neuron_contributions,
    plot_neuron_correlation_matrix,
    plot_neuron_graph,
    create_interactive_neuron_graph,
    plot_layer_analysis,
    create_neuron_graph
)
from utils.intervention import (
    calculate_perplexity,
    last_layer_intervention
)

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

    # %%
    # Analyzing the SVD vectors
    # Analyze contributions
    U, S, V = analyze_neuron_contributions(toxic_matrix, vector_ids)

    # Get and print top neurons per component
    components_info = get_top_neurons_per_component(U, vector_ids)
    print_component_analysis(components_info, model, toxic_matrix, vector_ids)

    
    # %%
    # Create visualization
    fig = plot_neuron_contributions(U, vector_ids, S)
    fig.savefig("./plots/neuron_contributions.png")
    
    
    # %%
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
    
    # Load RealToxicityPrompts dataset
    toxicity_prompts = load_dataset("allenai/real-toxicity-prompts", split="train")
    filtered_prompts = toxicity_prompts.filter(lambda x: x['continuation']['toxicity'] is not None and x['continuation']['toxicity'] > 0.9)


    
    # %%
    
    # %%
    from pprint import pprint
    pprint(last_layer_intervention(model, "You are such a piece of", top_15_info[1]['vector'], alpha=1, max_tokens=10, temperature=0.9))


    # %%
    import pandas as pd

    results = []
    for i, vector_info in enumerate(top_15_info):
        # Get intervention results for each vector
        result = last_layer_intervention(model, "You are such a piece of", vector_info['vector'], 
                                    alpha=1, max_tokens=10, temperature=0.9)
        
        # Extract relevant metrics
        result_dict = {
            'vector_index': i,
            'baseline_output': result['baseline_output'],
            'hooked_output': result['hooked_output'],
            'baseline_perplexity': result['baseline_perplexity'],
            'hooked_perplexity': result['hooked_perplexity'],
            'baseline_toxicity': result['baseline_toxic_metrics']['toxicity'],
            'hooked_toxicity': result['hooked_toxic_metrics']['toxicity'],
            'baseline_insult': result['baseline_toxic_metrics']['insult'],
            'hooked_insult': result['hooked_toxic_metrics']['insult'],
            'baseline_obscene': result['baseline_toxic_metrics']['obscene'],
            'hooked_obscene': result['hooked_toxic_metrics']['obscene']
        }
        results.append(result_dict)

    # Create DataFrame
    df_results = pd.DataFrame(results)

    # Calculate toxicity differences
    df_results['toxicity_increase'] = df_results['hooked_toxicity'] - df_results['baseline_toxicity']
    df_results['perplexity_change'] = df_results['hooked_perplexity'] - df_results['baseline_perplexity']

    # Sort by toxicity increase to see most impactful vectors
    df_results = df_results.sort_values('toxicity_increase', ascending=False)

    # Display the results
    print(df_results)

    # Optionally save to CSV
    df_results.to_csv('toxic_vector_results.csv', index=False)



# %%

if __name__ == "__main__":
    main()
# %%
