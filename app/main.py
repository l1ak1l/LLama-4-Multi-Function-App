from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.routes import chat, ocr, agents
from app.utils.temp_cleanup import TempCleanup
import asyncio

# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     # Start up
#     cleanup_task = asyncio.create_task(TempCleanup.start_cleanup_scheduler())
#     yield
#     # Shut down (if needed)
#     cleanup_task.cancel()

app = FastAPI(title="Llama-4 API")

# Include routers
app.include_router(chat.router)
app.include_router(ocr.router)
#app.include_router(rag.router)
app.include_router(agents.router)

