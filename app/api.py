"""
FastAPI application for the Shipping Guide Extraction Service.
"""


from contextlib import asynccontextmanager
import threading
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from app.agents.shipping.crew import run_shipping_crew
from app.config import settings
from app.models.shipping import ShippingExtractRequest, ShippingExtractResponse
from app.worker import run_worker_loop

# ── Lifespan (Worker Thread) ──────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Start the worker thread
    stop_event = threading.Event()
    worker_thread = threading.Thread(
        target=run_worker_loop, 
        args=(stop_event,), 
        daemon=True,
        name="AIOrderWorker"
    )
    worker_thread.start()
    
    yield
    
    # Shutdown: Signal worker to stop
    stop_event.set()
    worker_thread.join(timeout=5.0)


app = FastAPI(
    title="Punto Gafas Shipping Guide Extractor",
    description="AI-powered shipping guide extraction and order matching service.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for simplicity in this context
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "service": "shipping-guide-extractor"}


@app.post(
    "/api/v1/shipping-guides/extract",
    response_model=ShippingExtractResponse,
    status_code=status.HTTP_200_OK,
)
async def extract_shipping_guide(request: ShippingExtractRequest):
    """
    Extract data from a shipping guide image and match it to an order.
    
    This endpoint is called by the `whatsapp-media-processor` Edge Function
    when it detects a shipping guide image from a carrier.
    """
    try:
        # Run the CrewAI pipeline
        result = run_shipping_crew(request)
        return result
    except Exception as exc:
        # Log the error (it's already logged in run_shipping_crew, but just in case)
        # We return a failed response rather than 500 so the caller knows what happened
        return ShippingExtractResponse(
            success=False,
            error=str(exc),
        )
