import torch
from transformer_lens import HookedTransformer
from typing import Dict, Union, Optional, Callable
from datasets import load_dataset
from torch.nn.functional import cross_entropy
from .detoxify_utils import get_detoxify_scores

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
    
    Args:
        model: The transformer model
        dataset_name: Name of the dataset to use
        name: Specific configuration of the dataset
        split: Dataset split to use
        max_samples: Maximum number of samples to process
        hook_fn: Optional hook function to apply
        hook_point: Optional hook point to apply the hook
        
    Returns:
        float: Perplexity score
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
                          temperature: float = 0.7,
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
    original_output = model.generate(prompt, max_new_tokens=max_tokens, temperature=temperature)
    
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
    hooked_output = model.generate(prompt, max_new_tokens=max_tokens, temperature=temperature)
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