from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.models.llm_handler import FinanceRAG
import uuid

app = FastAPI()
model = FinanceRAG()

# Memory for conversation history (in RAM)
conversations = {}

class ChatRequest(BaseModel):
    message: str
    conversation_id: str = None

class ChatResponse(BaseModel):
    response: str
    conversation_id: str

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    # Create new conversation if needed
    if not request.conversation_id:
        conv_id = str(uuid.uuid4())
        conversations[conv_id] = []
    else:
        conv_id = request.conversation_id
        if conv_id not in conversations:
            raise HTTPException(status_code=404, detail="Invalid conversation ID")
    
    # Add context from previous messages
    context = "\n".join(conversations[conv_id][-3:])
    full_query = f"{context}\nUser: {request.message}" if context else request.message
    
    # Generate response
    response = model.generate(full_query)
    
    # Update conversation history
    conversations[conv_id].extend([
        f"User: {request.message}",
        f"Assistant: {response}"
    ])
    
    return ChatResponse(response=response, conversation_id=conv_id)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)