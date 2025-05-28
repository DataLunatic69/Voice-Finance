from langchain_groq import ChatGroq
import os

def get_llm():
    return ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="deepseek-r1-distill-llama-70b"
    )