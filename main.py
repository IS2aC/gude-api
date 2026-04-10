"""

╔══════════════════════════════════════════════════════════╗
║        WAHOU — API FastAPI du GUDE                       ║
║        Guichet Unique du Développement des PME           ║
║        Endpoint : POST /chat                             ║
╚══════════════════════════════════════════════════════════╝

"""
import os 
from models import ChatRequest, ChatResponse, ResetResponse, HealthResponse
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from fastapi.staticfiles import StaticFiles
from ai_services.main import ChatbotManager
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

# ✅ CORRECTION : utiliser __file__ au lieu de __name__
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))


# ──────────────────────────────────────────────
# 🚀 Initialisation FastAPI
# ──────────────────────────────────────────────
app = FastAPI(
    title="Wahou — API GUDE",
    description=(
        "API du chatbot Wahou, assistante virtuelle du "
        "Guichet Unique du Développement des PME (GUDE) en Côte d'Ivoire.\n\n"
        "Chaque utilisateur doit fournir un `session_id` unique pour maintenir "
        "l'historique de conversation."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
chatbot_manager: Optional[ChatbotManager] = None


@app.on_event("startup")
async def startup_event():
    global chatbot_manager
    print("⏳ Chargement de Wahou...")
    chatbot_manager = ChatbotManager()
    print("✅ Wahou est prête !")


# ──────────────────────────────────────────────
# 📡 Endpoints
# ──────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={}   # context optionnel si tu veux passer des variables à Jinja2
    )


@app.get("/health", response_model=HealthResponse, tags=["Santé"])
def health_check():
    if chatbot_manager is None:
        raise HTTPException(status_code=503, detail="Service non disponible")
    return HealthResponse(
        status="ok",
        service="Wahou — Assistante Virtuelle GUDE",
        version="1.0.0"
    )

@app.get("/test_chat", response_class=HTMLResponse)
async def test_chat(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="test_chat.html",
        context={}
    )

@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
def chat(request: ChatRequest):
    """
    ## Envoyer un message à Wahou

    - **session_id** : identifiant unique de session (ex: UUID généré côté client)
    - **message** : question ou message de l'utilisateur

    L'historique de conversation est conservé **30 minutes** par session.
    """
    if chatbot_manager is None:
        raise HTTPException(status_code=503, detail="Le service Wahou n'est pas disponible")

    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Le message ne peut pas être vide")

    if not request.session_id.strip():
        raise HTTPException(status_code=400, detail="Le session_id ne peut pas être vide")

    is_new = not chatbot_manager.session_exists(request.session_id)

    try:
        chat_engine = chatbot_manager.get_chat_engine(request.session_id)
        response = chat_engine.chat(request.message)

        return ChatResponse(
            session_id=request.session_id,
            message=request.message,
            response=str(response),
            is_new_session=is_new
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors du traitement de votre message : {str(e)}"
        )


@app.delete("/chat/{session_id}", response_model=ResetResponse, tags=["Chat"])
def reset_session(session_id: str):
    """
    ## Réinitialiser une session

    Supprime l'historique de conversation d'un utilisateur.
    """
    if chatbot_manager is None:
        raise HTTPException(status_code=503, detail="Le service Wahou n'est pas disponible")

    success = chatbot_manager.clear_session(session_id)

    return ResetResponse(
        session_id=session_id,
        success=success,
        message=(
            "Session réinitialisée avec succès." if success
            else "Aucune session active trouvée pour cet identifiant."
        )
    )


# ──────────────────────────────────────────────
# 🏃 Lancement direct
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)