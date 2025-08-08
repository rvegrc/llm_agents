import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel

load_dotenv()

app = FastAPI()

API_TOKEN = os.getenv("API_TOKEN")  # your secret token set in .env

class PromptRequest(BaseModel):
    prompt: str
    user_id: str
    thread_id: str

@app.post("/generate")
async def generate_text(request: PromptRequest):
    from langgraph_agent import chat_with_agent
    try:
        generated_text = chat_with_agent(request.prompt, request.user_id, request.thread_id)
        return {"generated_text": generated_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/test")  # use POST for requests with JSON body
async def generate_text(request: PromptRequest):
    # Convert user_id to string before concatenation or add as integer math
    new_prompt = request.prompt + " - modified"
    new_user_id = str(request.user_id) + "11111"
    return {
        "origin id": request.user_id,
        "new id": new_user_id,
        "origin prompt": request.prompt,
        "new prompt": new_prompt
    }


@app.get("/")
def read_root():
    return {"Hello": "This is the root of the API"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run('api:app', host="0.0.0.0", port=8000, reload=True)
