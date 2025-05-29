"""
Main FastAPI application for agents microservices
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .api import router

app = FastAPI(
    title="Financial Analysis Agents API",
    description="Microservices API for all financial analysis agents",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include agent routes
app.include_router(router, prefix="/agents", tags=["agents"])

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Financial Analysis Agents API",
        "version": "1.0.0",
        "available_endpoints": [
            "/agents/speech_input",
            "/agents/extract_keywords", 
            "/agents/fetch_price_data",
            "/agents/fetch_news_data",
            "/agents/retrieve_price_docs",
            "/agents/retrieve_news_docs",
            "/agents/analyze_price",
            "/agents/analyze_news",
            "/agents/synthesize_report",
            "/agents/health"
        ]
    }

@app.get("/health")
async def health_check():
    """Main health check endpoint"""
    return {"status": "healthy", "service": "agents_microservice_main"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)