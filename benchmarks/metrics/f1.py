"""Token-level F1 Score Calculation."""
import re

def tokenize(text: str) -> set[str]:
    """Simple whitespace/punctuation tokenizer."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return set(text.split())

def calculate_f1(prediction: str, reference: str) -> float:
    """Calculate token-level F1 score."""
    pred_tokens = tokenize(prediction)
    ref_tokens = tokenize(reference)
    
    if not pred_tokens or not ref_tokens:
        return 0.0
    
    common = pred_tokens & ref_tokens
    
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(ref_tokens)
    
    if precision + recall == 0:
        return 0.0
    
    return 2 * precision * recall / (precision + recall)
