import os
import time
import logging
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from dotenv import load_dotenv
from openai import AzureOpenAI
import posthog

# ------------------------------------------------------------
# 1️⃣ Environment setup
# ------------------------------------------------------------
load_dotenv()

# Configure PostHog
POSTHOG_KEY = os.getenv("POSTHOG_API_KEY")
POSTHOG_HOST = os.getenv("POSTHOG_HOST", "https://us.i.posthog.com")

if POSTHOG_KEY:
    posthog.project_api_key = POSTHOG_KEY
    posthog.host = POSTHOG_HOST
else:
    print("⚠️  PostHog not configured (missing POSTHOG_API_KEY)")

# ------------------------------------------------------------
# 3️⃣ FastAPI & Rate Limiter setup
# ------------------------------------------------------------
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(title="ShadowClone AI – Portfolio Assistant")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Friendly custom message for rate limit exceeded
@app.exception_handler(RateLimitExceeded)
async def custom_rate_limit_handler(request, exc):
    return JSONResponse(
        status_code=429,
        content={
            "error": "Too many requests. Please slow down — ShadowClone AI needs a breather 🤖💨"
        },
    )

# ------------------------------------------------------------
# 4️⃣ CORS configuration
# ------------------------------------------------------------
origins = [o.strip() for o in os.getenv("CORS_ALLOW_ORIGINS", "*").split(",")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------
# 5️⃣ Azure OpenAI setup
# ------------------------------------------------------------
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
)

ASSISTANT_ID = os.getenv("SHADOWCLONE_ASSISTANT_ID")
if not ASSISTANT_ID:
    raise ValueError("❌ SHADOWCLONE_ASSISTANT_ID not set in environment.")

# ------------------------------------------------------------
# 6️⃣ Routes
# ------------------------------------------------------------
@app.get("/health")
def health():
    """Simple health check endpoint."""
    return {"status": "ok"}

@app.get("/ping")
def ping():
    """
    Lightweight endpoint used by the frontend to wake up the Render container.
    """
    return {"message": "pong"}


@app.post("/ask")
@limiter.limit("10/minute;60/hour")
async def ask(request: Request):
    data = await request.json()
    question = data.get("question", "").strip()
    thread_id = data.get("thread_id")

    if not question:
        return JSONResponse({"error": "No question provided."}, status_code=400)

    logging.info(f"USER QUESTION: {question}")

    # 1️⃣ Create or reuse thread
    if not thread_id:
        thread = client.beta.threads.create()
        thread_id = thread.id
        logging.info(f"🧠 New thread created: {thread_id}")
    else:
        logging.info(f"Using existing thread: {thread_id}")

    # 2️⃣ Add user message
    client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=question
    )

    # 3️⃣ Run the ShadowClone AI assistant
    run = client.beta.threads.runs.create(
        thread_id=thread_id,
        assistant_id=ASSISTANT_ID
    )

    # 4️⃣ Poll until complete
    while run.status in ["queued", "in_progress"]:
        time.sleep(1)
        run = client.beta.threads.runs.retrieve(
            thread_id=thread_id,
            run_id=run.id
        )

    # 5️⃣ Retrieve message
    if run.status == "completed":
        messages = client.beta.threads.messages.list(thread_id=thread_id)
        answer = messages.data[0].content[0].text.value
        logging.info(f"ASSISTANT REPLY:\n{answer}\n{'-'*80}")
        return {"answer": answer, "thread_id": thread_id}

    logging.error(f"Run failed with status: {run.status}")
    # ✅ Log to PostHog
    try:
        posthog.capture(
            distinct_id=request.client.host,  # user IP
            event="shadowclone_question",
            properties={
                "question": question,
                "latency_sec": 2,
                "status": run.status,
                "thread_id": thread_id,
            },
        )
    except Exception as e:
        logging.error(f"PostHog logging failed: {e}")

    return JSONResponse(
        {"error": f"Run failed with status: {run.status}", "thread_id": thread_id},
        status_code=500,
    )

