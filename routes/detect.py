import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse

from services.ai_service import detect_defects, AiTimeoutError, AiParseError

router = APIRouter()

ALLOWED_MIME_TYPES = {"image/jpeg", "image/png"}
MAX_FILE_BYTES = 10 * 1024 * 1024  # 10 MB


@router.post("/detect")
async def detect_endpoint(image: UploadFile = File(...)):
    # Validate MIME type
    if image.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Only JPEG and PNG images are supported",
        )

    data = await image.read()

    # Validate size (after read so we have the true byte count)
    if len(data) > MAX_FILE_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds 10 MB server limit")

    try:
        detections = await detect_defects(data, image.content_type)
    except AiTimeoutError:
        raise HTTPException(
            status_code=504,
            detail="AI analysis timed out — please try again",
        )
    except AiParseError as exc:
        raise HTTPException(status_code=500, detail=f"AI response parse error: {exc}")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Detection failed: {exc}")

    return JSONResponse(
        content={
            "sessionId": str(uuid.uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "detections": detections,
        }
    )
