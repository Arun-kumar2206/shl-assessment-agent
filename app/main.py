from fastapi import FastAPI
from fastapi.responses import JSONResponse

from .agent import AssessmentAgent
from .schema import ChatRequest, ChatResponse

app = FastAPI()
agent: AssessmentAgent | None = None


@app.on_event("startup")
def startup() -> None:
    global agent
    if agent is None:
        agent = AssessmentAgent("data/raw/shl_product_catalog.json")


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if agent is None:
        return JSONResponse(status_code=503, content={"error": "Service not ready"})
    payload = agent.respond([msg.model_dump() for msg in request.messages])
    return ChatResponse(**payload)
