from fastapi import FastAPI, Request
from pydantic import BaseModel
from langgraph_agent import build_agent

app = FastAPI()
agent = build_agent()

class UserInput(BaseModel):
    message: str
    session_id: str  # Optional: for multi-user memory

@app.post("/chat")
def chat(user_input: UserInput):
    state = {"input": user_input.message, "session_id": user_input.session_id}
    result = agent.invoke(state)
    return {"response": result.get("input", "Sorry, something went wrong.")}