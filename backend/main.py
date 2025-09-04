import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from endpoints import router as api_router
from fastapi.staticfiles import StaticFiles
from functions import ASSETS_DIR

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_ngrok_skip_header(request, call_next):
    response = await call_next(request)
    response.headers["ngrok-skip-browser-warning"] = "true"
    return response
@app.middleware("http")
async def add_custom_headers(request, call_next):
    response = await call_next(request)

    response.headers["ngrok-skip-browser-warning"] = "true"
    response.headers["Access-Control-Expose-Headers"] = "X-Sources"
    response.headers["Access-Control-Allow-Origin"] = "*"

    if request.url.path.startswith("/assets/"):
        # Allow embedding
        response.headers["X-Frame-Options"] = "ALLOWALL"
        response.headers["Content-Security-Policy"] = "frame-ancestors *"

        # Fix MIME issues
        if request.url.path.endswith(".pdf"):
            response.headers["Content-Type"] = "application/pdf"
            response.headers["Cache-Control"] = "no-cache"

    return response


# mount static assets so frontend can fetch /assets/...
os.makedirs(ASSETS_DIR, exist_ok=True)
app.mount("/assets", StaticFiles(directory=ASSETS_DIR), name="assets")

app.include_router(api_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
