import os
import logging
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Header, logger
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from langgraph_agent import chat_with_agent

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

logging.getLogger().info("Logging is set up.")


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

API_TOKEN = os.getenv("API_TOKEN")
REQUIRE_API_TOKEN = os.getenv("REQUIRE_API_TOKEN", "false").lower() == "true"

# Request model
class PromptRequest(BaseModel):
    prompt: str
    user_id: str
    thread_id: str

@app.post("/generate")
async def generate_text(
    request: PromptRequest,
    authorization: str | None = Header(default=None)
):
    print("Endpoint /generate called") 
    logging.info(f"Incoming /generate request payload: {request.model_dump_json()}") 
    # logging.info(f"Incoming /generate request | user_id={request.user_id} thread_id={request.thread_id}")

    if REQUIRE_API_TOKEN:
        if not authorization:
            logging.warning("Unauthorized: Missing Authorization header")
            raise HTTPException(status_code=401, detail="You must provide an API token in the Authorization header.")
        
        if not authorization.startswith("Bearer "):
            logging.warning("Unauthorized: Invalid Authorization header format")
            raise HTTPException(status_code=400, detail="Authorization header must be in format: Bearer <API_TOKEN>.")

        token = authorization.removeprefix("Bearer ").strip()
        if token != API_TOKEN:
            logging.warning("Unauthorized: Invalid API token")
            raise HTTPException(status_code=403, detail="Invalid API token. Please provide the correct one.")

    try:
        generated_text = chat_with_agent(request.prompt, request.user_id, request.thread_id)
        logging.info("Generated text successfully")
        return {"generated_text": generated_text}

    except Exception as e:
        logging.exception("Error in /generate endpoint")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.get("/")
def read_root():
    return {"Hello": "This is the root of the API"}

if __name__ == "__main__":
    uvicorn.run('api:app', host="0.0.0.0", port=8000, reload=True)