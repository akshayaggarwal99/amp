import google.generativeai as genai
import os

API_KEY = __import__("os").environ.get("GEMINI_API_KEY") or (_ for _ in ()).throw(RuntimeError("GEMINI_API_KEY not set"))
genai.configure(api_key=API_KEY)

print("Listing available models...")
try:
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(m.name)
except Exception as e:
    print(f"Error listing models: {e}")
