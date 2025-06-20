from fastapi import FastAPI
from app.schemas import ChatRequest, ChatResponse
from app.model_service import ChatService

app = FastAPI(title="LabChatbot API")
service = ChatService()


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    reply = service.generate(req.text)
    return ChatResponse(reply=reply)