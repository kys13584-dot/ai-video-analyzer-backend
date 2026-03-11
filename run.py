import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        workers=4,      # Allow multiple requests while background tasks run
        reload=False,   # reload=True is incompatible with workers > 1
    )
