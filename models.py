from pydantic import BaseModel

# ──────────────────────────────────────────────
# 📦 Modèles Pydantic (Request / Response)
# ──────────────────────────────────────────────
class ChatRequest(BaseModel):
    session_id: str
    message: str

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "session_id": "user-abc-123",
                    "message": "C'est quoi le GUDE ?"
                }
            ]
        }
    }


class ChatResponse(BaseModel):
    session_id: str
    message: str
    response: str
    is_new_session: bool


class ResetResponse(BaseModel):
    session_id: str
    success: bool
    message: str


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str