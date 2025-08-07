import requests
import os
import re
from dotenv import load_dotenv
load_dotenv()

LLM_API_SERVER_URL = os.getenv("LLM_API_SERVER_URL")

# model = "qwen3:0.6b"

model = "deepseek-r1:1.5b"  # Example model, change as needed

def llm_chat_tool(messages):
    """Send messages to LLM API and return the response"""
    try:
        response = requests.post(
            url=LLM_API_SERVER_URL,
            headers={"Content-Type": "application/json"},
            json={
                "model": model,
                "messages": messages,
                "temperature": 0.5,
                "max_tokens": 256
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