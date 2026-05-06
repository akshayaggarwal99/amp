"""BLEU-1 Score Calculation."""
import re

def tokenize(text: str) -> list[str]:
    """Simple whitespace/punctuation tokenizer."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text.split()

def calculate_bleu(prediction: str, reference: str) -> float:
    """Calculate BLEU-1 (unigram overlap) score."""
    pred_tokens = tokenize(prediction)
    ref_tokens = tokenize(reference)
    
    if not pred_tokens or not ref_tokens:
        return 0.0
    
    # Count matching tokens
    matches = sum(1 for token in pred_tokens if token in ref_tokens)
    
    # Precision-based BLEU-1
    return matches / len(pred_tokens) if pred_tokens else 0.0
