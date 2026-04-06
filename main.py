import base64
import os
import secrets
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

load_dotenv()

from routes.detect import router as detect_router  # noqa: E402  (after load_dotenv)

# ---------- Paths ----------
BASE_DIR   = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "html"          # built React app lives here

# ---------- Basic-auth config (from .env) ----------
_APP_USER = os.getenv("APP_USERNAME", "admin")
_APP_PASS = os.getenv("APP_PASSWORD", "changeme")
_AUTH_ENABLED = os.getenv("AUTH_ENABLED", "true").lower() not in ("false", "0", "no")

# Paths that bypass auth (health check only)
_NO_AUTH_PREFIXES = ("/api/health",)


def _check_basic_auth(authorization: str | None) -> bool:
    """Return True when the Authorization header contains valid Basic credentials."""
    if not authorization or not authorization.lower().startswith("basic "):
        return False
    try:
        decoded = base64.b64decode(authorization[6:]).decode("utf-8", errors="replace")
        user, _, pwd = decoded.partition(":")
        return secrets.compare_digest(user, _APP_USER) and secrets.compare_digest(pwd, _APP_PASS)
    except Exception:
        return False


@asynccontextmanager
async def lifespan(_app: FastAPI):
    yield


app = FastAPI(
    title="SnagAI",
    version="1.0.0",
    lifespan=lifespan,
    # Hide docs behind auth — they are disabled in production
    docs_url=None,
    redoc_url=None,
)


# ---------- Basic-auth middleware ----------
@app.middleware("http")
async def basic_auth_middleware(request: Request, call_next):
    if not _AUTH_ENABLED:
        return await call_next(request)

    # Skip auth for health check
    for prefix in _NO_AUTH_PREFIXES:
        if request.url.path.startswith(prefix):
            return await call_next(request)

    if not _check_basic_auth(request.headers.get("Authorization")):
        return Response(
            content="Unauthorised",
            status_code=401,
            headers={
                "WWW-Authenticate": "Basic realm=\"SnagAI\"",
                "Cache-Control": "no-store",
            },
        )
    return await call_next(request)


# ---------- CORS (still useful for local dev / API clients) ----------
_frontend_url = os.getenv("FRONTEND_URL", "http://localhost:5173")
_extra = [o.strip() for o in os.getenv("EXTRA_CORS_ORIGINS", "").split(",") if o.strip()]
allowed_origins = list({
    _frontend_url,
    "http://localhost:5173",
    "http://localhost:3001",
    *_extra,
})

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_origin_regex=(
        r"https?://[^/]+\.ngrok(-free)?\.app"
        r"|https?://[^/]+\.ngrok\.io"
        r"|https?://[^/]+\.mintbig\.com"
    ),
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    allow_credentials=True,
)

# ---------- API routers ----------
app.include_router(detect_router, prefix="/api")


# ---------- Health (no auth) ----------
@app.get("/api/health")
def health():
    return {"status": "ok", "ts": datetime.now(timezone.utc).isoformat()}


# ---------- Serve built React SPA ----------
if STATIC_DIR.exists():
    # Serve static assets (JS, CSS, images) at their exact paths
    app.mount("/assets", StaticFiles(directory=STATIC_DIR / "assets"), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def spa_fallback(full_path: str):
        """Return index.html for any non-API path so React Router works."""
        # Exact file match (favicon, manifest, etc.)
        candidate = STATIC_DIR / full_path
        if candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/", include_in_schema=False)
    async def spa_root():
        return FileResponse(STATIC_DIR / "index.html")
else:
    @app.get("/", include_in_schema=False)
    async def no_frontend():
        return HTMLResponse(
            "<h2>SnagAI API is running.</h2>"
            "<p>Build the frontend and copy <code>dist/</code> to <code>backend/html/</code>.</p>"
        )


# ---------- Generic 404 ----------
@app.exception_handler(404)
async def not_found(_req, _exc):
    # API 404s get JSON; everything else falls through to SPA
    if _req.url.path.startswith("/api/"):
        return JSONResponse(status_code=404, content={"error": "Not found"})
    if STATIC_DIR.exists():
        return FileResponse(STATIC_DIR / "index.html")
    return JSONResponse(status_code=404, content={"error": "Not found"})
