import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from endpoints import router as api_router


app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["localhost", "http://localhost:8501", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)