from detoxify import Detoxify

def get_detoxify_scores(text: str) -> dict:
    """Get toxicity scores using Detoxify model"""
    model = Detoxify('original')
    scores = model.predict(text)
    return scores

def print_warning():
    """Print warning message as in paper."""
    warning = """
WARNING: THESE EXAMPLES ARE HIGHLY OFFENSIVE.
We note that SVD.U_Toxic[2] has a particularly gendered nature.
This arises from the dataset and language model we use.
    """
    print(warning) 