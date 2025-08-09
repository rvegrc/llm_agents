import os
import logging
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("app.log"),          # File output
        logging.StreamHandler()                  # Console output
    ],
    force=True  # This overrides any prior logging config
)
logger = logging.getLogger(__name__)



# FastAPI app
app = FastAPI(
    title="LLM Agents API",
    description="Backend API for TG bot with LLM agent and tools",
    version="1.0.0"
)

# # CORS middleware
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # You can restrict to specific domains if needed
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

API_TOKEN = os.getenv("API_TOKEN")  # Optional: for authentication

# Request model
class PromptRequest(BaseModel):
    prompt: str
    user_id: str
    thread_id: str

@app.post("/generate")
async def generate_text(request: PromptRequest):
    """
    Generate a response from the LLM agent using the provided prompt, user_id, and thread_id.
    """
    logger.info(f"Incoming /generate request | user_id={request.user_id} thread_id={request.thread_id}")

    # Optional: check API token (if you want security for internal calls)
    if API_TOKEN and os.getenv("REQUIRE_API_TOKEN", "false").lower() == "true":
        # Here you could verify an Authorization header
        pass

    try:
        from langgraph_agent import chat_with_agent
        generated_text = chat_with_agent(request.prompt, request.user_id, request.thread_id)
        logger.info("Generated text successfully")
        return {"generated_text": generated_text}

    except Exception as e:
        logger.exception("Error in /generate endpoint")
        raise HTTPException(status_code=500, detail=str(e))

# @app.post("/generate")
# async def generate_text(request: PromptRequest):
#     return {"generated_text": f"Echo: {request.prompt}"}


@app.get("/health")
async def health_check():
    """
    Simple health check endpoint.
    """
    return {"status": "ok"}

@app.get("/")
def read_root():
    return {"Hello": "This is the root of the API"}


if __name__ == "__main__":
    uvicorn.run('api:app', host="0.0.0.0", port=8000, reload=True)
