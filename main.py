from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from config import ALLOWED_ORIGINS, PORT, API_SECRET_KEY
from routes.health import router as health_router
from routes.enrollment import router as enrollment_router
from routes.verify import router as verify_router

app = FastAPI(
    title="STU Face API",
    description="Face recognition API for teacher attendance (DeepFace + Anti-Spoofing)",
    version="1.0.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# API Key authentication middleware
@app.middleware("http")
async def verify_api_key(request: Request, call_next):
    # Skip auth for health check and docs
    if request.url.path in ("/health", "/docs", "/openapi.json", "/redoc"):
        return await call_next(request)

    if not API_SECRET_KEY:
        # No key configured = reject all (force setup)
        return JSONResponse(
            status_code=500,
            content={"detail": "API_SECRET_KEY not configured on server"},
        )

    api_key = request.headers.get("X-API-Key")
    if api_key != API_SECRET_KEY:
        return JSONResponse(
            status_code=401,
            content={"detail": "Invalid or missing API key"},
        )

    return await call_next(request)


# Routes
app.include_router(health_router, tags=["Health"])
app.include_router(enrollment_router, tags=["Enrollment"])
app.include_router(verify_router, tags=["Verification"])


@app.on_event("startup")
async def startup():
    """Pre-load DeepFace models on startup for faster first request."""
    print("🔄 Pre-loading DeepFace models...")
    try:
        from deepface import DeepFace
        from config import FACE_MODEL, FACE_DETECTOR
        import numpy as np

        # Create a dummy image to trigger model download/load
        dummy = np.zeros((224, 224, 3), dtype=np.uint8)
        dummy[50:150, 50:150] = 200  # light square
        try:
            DeepFace.represent(
                img_path=dummy,
                model_name=FACE_MODEL,
                detector_backend=FACE_DETECTOR,
                enforce_detection=False,
            )
        except Exception:
            pass  # Model is loaded even if detection fails on dummy
        print(f"✅ Models loaded: {FACE_MODEL} + {FACE_DETECTOR}")
    except Exception as e:
        print(f"⚠️ Model pre-load warning: {e}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=True)
