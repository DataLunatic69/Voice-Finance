from langchain_groq import ChatGroq
import os

def get_llm():
    # Using llama-3.3-70b-versatile - most powerful available model
    return ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.3-70b-versatile",
        temperature=0.3,  # Lower temperature for more consistent analysis
        max_tokens=2048
    )