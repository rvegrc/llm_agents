import requests
import os
import re
from typing import List
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.llms import HuggingFacePipeline

from dotenv import load_dotenv
load_dotenv()

LLM_API_SERVER_URL = os.getenv("LLM_API_SERVER_URL")

# model = "qwen3:0.6b"

model = os.getenv("LLM_MODEL_NAME")  # Default to deepseek-r1:1.5b if not set

def llm_chat_tool(messages) -> str:
    """Send messages to LLM API and return the response"""
    try:
        response = requests.post(
            url=LLM_API_SERVER_URL,
            headers={"Content-Type": "application/json"},
            json={
                "model": model,
                "messages": messages,
                "temperature": 0.6,
                "max_tokens": 500
            }
        )
        response.raise_for_status()
        raw_content = response.json()["choices"][0]["message"]["content"]
        
        # Remove <think>...</think> blocks
        clean_content = re.sub(r'<think>.*?</think>', '', raw_content, flags=re.DOTALL)
        
        return clean_content.strip()

    except requests.exceptions.RequestException as e:
        raise Exception(f"API request failed: {str(e)}")
    except (KeyError, IndexError) as e:
        raise Exception(f"Unexpected API response format: {str(e)}")
    
def llm_call(local_model_path: str, model) -> HuggingFacePipeline:
    # Load from local path
    tokenizer = AutoTokenizer.from_pretrained(local_model_path)
    model = AutoModelForCausalLM.from_pretrained(
        local_model_path,
        device_map="auto",
        torch_dtype="auto"
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512
    )

    # Wrap for LangGraph
    return HuggingFacePipeline(pipeline=pipe)