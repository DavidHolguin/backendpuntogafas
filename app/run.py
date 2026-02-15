"""
Entry point to run the FastAPI API server.

Usage:
    python -m app.run
"""

import uvicorn
from app.config import settings

def main():
    """Run the uvicorn server."""
    uvicorn.run(
        "app.api:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True  # useful for dev
    )

if __name__ == "__main__":
    main()
