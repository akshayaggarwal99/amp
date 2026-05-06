"""LLM-as-Judge Evaluation using Gemini."""
import google.generativeai as genai
import json
import re

# API Key (passed from harness)
_model = None

def _get_model(api_key: str):
    global _model
    if _model is None:
        genai.configure(api_key=api_key)
        _model = genai.GenerativeModel('gemini-2.5-flash-lite')
    return _model

JUDGE_PROMPT = """
Your task is to label an answer as CORRECT or WRONG.

Question: {question}
Ground Truth: {gold}
Predicted: {predicted}

Rules:
- Be GENEROUS. If the prediction touches on the same topic, mark CORRECT.
- For time questions, "last Tuesday" matching "The Tuesday before Oct 15" is CORRECT.
- Format differences (e.g., "May 7th" vs "7 May") are CORRECT.

Respond with JSON: {{"label": "CORRECT"}} or {{"label": "WRONG"}}
"""

def llm_judge(question: str, gold: str, predicted: str, api_key: str) -> int:
    """Use LLM to judge if prediction is correct. Returns 1 or 0."""
    model = _get_model(api_key)
    
    prompt = JUDGE_PROMPT.format(question=question, gold=gold, predicted=predicted)
    
    try:
        response = model.generate_content(prompt)
        text = response.text.strip()
        
        # Extract JSON
        match = re.search(r'\{.*?\}', text, re.DOTALL)
        if match:
            data = json.loads(match.group())
            return 1 if data.get("label", "").upper() == "CORRECT" else 0
        
        # Fallback: check for CORRECT/WRONG in text
        if "CORRECT" in text.upper():
            return 1
        return 0
    except Exception as e:
        print(f"LLM Judge Error: {e}")
        return 0
