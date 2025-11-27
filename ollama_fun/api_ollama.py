from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from pydantic import BaseModel, Field
from typing import Optional
import ollama
import uvicorn

app = FastAPI(
    title="Ollama Chat API",
    description="API do generowania odpowiedzi za pomocą modelu Ollama",
    version="1.0.0"
)


class ChatRequest(BaseModel):
    """Model żądania dla endpointu chat"""
    content: str = Field(..., description="Treść wiadomości do modelu")
    model: str = Field(default="gemma3:12b", description="Nazwa modelu Ollama")
    temperature: float = Field(default=2.0, ge=0.0, le=2.0, description="Kreatywność modelu (0.0-2.0)")
    top_p: float = Field(default=0.7, ge=0.0, le=1.0, description="Nucleus sampling")
    top_k: int = Field(default=40, ge=1, description="Top-k sampling")
    num_predict: int = Field(default=100, ge=1, description="Maksymalna długość odpowiedzi")


class ChatResponse(BaseModel):
    """Model odpowiedzi z endpointu chat"""
    content: str
    model: str
    success: bool


@app.get("/")
async def root():
    """Endpoint główny"""
    return {
        "message": "Witaj w Ollama Chat API",
        "endpoints": {
            "/chat": "POST - Wyślij wiadomość do modelu",
            "/chat-with-image": "POST - Analizuj obraz za pomocą modelu vision",
            "/models": "GET - Lista dostępnych modeli",
            "/health": "GET - Status serwisu",
            "/docs": "GET - Dokumentacja API"
        }
    }


@app.get("/models")
async def list_models():
    """Endpoint zwracający listę dostępnych modeli Ollama"""
    try:
        response = ollama.list()
        # Konwertuj obiekt ListResponse na słownik
        models_data = response.model_dump() if hasattr(response, 'model_dump') else {'models': []}

        # Wyciągnij nazwy modeli
        model_names = []
        if 'models' in models_data:
            for model in models_data['models']:
                if isinstance(model, dict):
                    model_names.append(model.get('model', model.get('name', 'unknown')))
                else:
                    # Jeśli to obiekt, spróbuj pobrać atrybut
                    model_names.append(getattr(model, 'model', getattr(model, 'name', str(model))))

        return {
            "success": True,
            "models": model_names,
            "count": len(model_names)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Błąd pobierania listy modeli: {str(e)}")


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Endpoint do generowania odpowiedzi za pomocą modelu Ollama

    Args:
        request: Obiekt zawierający parametry żądania

    Returns:
        ChatResponse: Odpowiedź z modelu
    """
    try:
        response = ollama.chat(
            model=request.model,
            messages=[{'role': 'user', 'content': request.content}],
            options={
                'temperature': request.temperature,
                'top_p': request.top_p,
                'top_k': request.top_k,
                'num_predict': request.num_predict,
            }
        )

        return ChatResponse(
            content=response['message']['content'],
            model=request.model,
            success=True
        )

    except ollama.ResponseError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Błąd modelu Ollama: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Nieoczekiwany błąd: {str(e)}"
        )


@app.post("/chat-with-image", response_model=ChatResponse)
async def chat_with_image(
    image: UploadFile = File(..., description="Plik obrazu do analizy"),
    content: str = Form(..., description="Pytanie dotyczące obrazu"),
    model: str = Form(default="llava", description="Nazwa modelu Ollama z obsługą vision"),
    temperature: float = Form(default=0.7, ge=0.0, le=2.0),
    top_p: float = Form(default=0.9, ge=0.0, le=1.0),
    top_k: int = Form(default=40, ge=1),
    num_predict: int = Form(default=200, ge=1)
):
    """
    Endpoint do analizy obrazów za pomocą modelu Ollama z obsługą vision

    Args:
        image: Plik obrazu (JPG, PNG, itp.)
        content: Pytanie lub opis tego, co chcesz wiedzieć o obrazie
        model: Nazwa modelu (domyślnie llava - model z obsługą vision)
        temperature: Kreatywność modelu (0.0-2.0)
        top_p: Nucleus sampling (0.0-1.0)
        top_k: Top-k sampling (≥1)
        num_predict: Maksymalna długość odpowiedzi (≥1)

    Returns:
        ChatResponse: Odpowiedź z analizy obrazu
    """
    try:
        # Sprawdź typ pliku
        if not image.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400,
                detail=f"Nieprawidłowy typ pliku. Oczekiwano obrazu, otrzymano: {image.content_type}"
            )

        # Wczytaj zawartość obrazu
        image_bytes = await image.read()

        # Wywołaj model Ollama z obrazem
        response = ollama.chat(
            model=model,
            messages=[{
                'role': 'user',
                'content': content,
                'images': [image_bytes]
            }],
            options={
                'temperature': temperature,
                'top_p': top_p,
                'top_k': top_k,
                'num_predict': num_predict,
            }
        )

        return ChatResponse(
            content=response['message']['content'],
            model=model,
            success=True
        )

    except ollama.ResponseError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Błąd modelu Ollama: {str(e)}. Upewnij się, że używasz modelu z obsługą vision (np. llava, llava:13b, bakllava)."
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Nieoczekiwany błąd: {str(e)}"
        )


@app.get("/health")
async def health_check():
    """Endpoint do sprawdzania statusu serwisu"""
    try:
        # Sprawdź czy Ollama działa
        ollama.list()
        return {
            "status": "healthy",
            "ollama_connection": "ok"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "ollama_connection": "error",
            "error": str(e)
        }


if __name__ == "__main__":
    # Uruchom serwer
    uvicorn.run(
        "api_ollama:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
