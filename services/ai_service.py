"""
ai_service.py — THE ONLY module that imports the OpenAI SDK.

All AI calls go through detect_defects(). Everything else in the project is
forbidden from importing openai directly.
"""
import asyncio
import base64
import json
import uuid

from openai import AsyncOpenAI

client = AsyncOpenAI()  # reads OPENAI_API_KEY from environment automatically

# ── Custom exceptions ────────────────────────────────────────────────────────

class AiTimeoutError(Exception):
    """Raised when the OpenAI call exceeds the allowed timeout."""


class AiParseError(Exception):
    """Raised when the model response cannot be parsed into the expected shape."""


# ── Taxonomy (matches client spec exactly) ───────────────────────────────────

VALID_DEFECT_TYPES = {
    "paint_finish_defect",        # peeling, bubbling, cracking, uneven, patchy paint
    "tile_misalignment",          # lippage, uneven grout lines, offset tiles
    "cracked_damaged_tiles",      # cracked, chipped, broken tiles on floor/wall
    "joinery_defect",             # doors/windows/cabinets misaligned, gaps, damage
    "sealant_silicone_defect",    # missing, cracked, discoloured, poorly applied sealant
    "incomplete_mep",             # exposed / incomplete mechanical, electrical, plumbing
    "general_surface_damage",     # scratches, dents, scuffs, marks on any surface
    "ceiling_defect",             # cracks, sagging, holes, water marks, damage to ceiling
    "damp_water_stain",           # damp patches, water stains, moisture marks
    "other",                      # anything that doesn't fit the above categories
}

SYSTEM_PROMPT = """You are an expert construction snagging and defect inspection AI.
You have 20 years of experience inspecting newly built and renovated properties for
handover quality issues (snags).

Your job is to examine the provided photograph and identify EVERY visible defect,
quality issue, or non-conformance — no matter how minor. The image may show:
  - Interior rooms (walls, ceilings, floors, doors, windows)
  - Bathrooms, kitchens (tiles, sealant, fixtures)
  - Building corridors, lobbies, common areas
  - Exterior elements, outdoor construction sites
  - Any stage from structural works to final fit-out

You MUST classify every defect into one of these categories:

1. paint_finish_defect — Paint Finish Defects
   Look for: peeling, bubbling, flaking, cracking, uneven coverage, roller marks,
   brush marks, drips, runs, missed areas, colour inconsistency, poor cutting-in,
   paint on wrong surfaces, patchy finish, touch-up marks visible.

2. tile_misalignment — Tile Misalignment / Lippage
   Look for: uneven tile edges (lippage), inconsistent grout line width, tiles not
   level, offset patterns, tiles not plumb, grout missing or crumbling.

3. cracked_damaged_tiles — Cracked / Damaged Tiles
   Look for: cracked tiles, chipped edges, broken tiles, scratched tile surfaces,
   holes in tiles, hairline cracks.

4. joinery_defect — Joinery Damage / Misalignment
   Look for: doors not aligned, gaps around door/window frames, cabinet doors
   misaligned, drawer fronts uneven, skirting board gaps, architrave gaps or damage,
   scratched or dented woodwork, poorly fitted shelving, loose handles/hinges.

5. sealant_silicone_defect — Sealant / Silicone Defects
   Look for: missing sealant, cracked or split sealant, discoloured silicone,
   poorly tooled sealant lines, gaps where sealant should be (around baths, showers,
   sinks, worktops, windows), mould in sealant.

6. incomplete_mep — Exposed / Incomplete MEP Installation
   Look for: exposed wiring, junction boxes without covers, pipes not boxed in,
   missing switch/socket plates, unfinished plumbing connections, exposed ductwork,
   cables trailing across surfaces, missing radiator covers.

7. general_surface_damage — General Surface Damage
   Look for: scratches, dents, scuffs on walls/floors/doors/worktops, marks left
   by trades, stickers/labels not removed, protective film not removed, damage to
   glass, chips in worktops or surfaces.

8. ceiling_defect — Ceiling Defects
   Look for: cracks in ceiling, sagging ceiling sections, holes, water marks on
   ceiling, uneven plasterwork, visible joints in plasterboard, missing ceiling
   tiles, poorly fitted cornices or coving, exposed rafters/joists where there
   should be a finished ceiling.

9. damp_water_stain — Damp / Water Stains
   Look for: damp patches on walls/ceiling/floor, water stain marks, tide marks,
   discolouration from moisture, mould or mildew growth, condensation damage.

10. other — Anything that does not fit categories 1-9 (e.g. safety hazards,
    exposed rebar, structural issues on construction sites, debris/waste).

RULES:
- Be THOROUGH. Identify every distinct issue. Do NOT group separate issues into one.
- If the SAME defect type appears in more than one physical location in the image,
  create a SEPARATE detection entry for EACH location with its own unique boundingBox.
  Example: two areas of peeling paint = TWO separate paint_finish_defect entries.
  NEVER merge multiple physical instances into a single detection.
- Almost every construction/property photo will have multiple issues -- return all of them.
- For each issue found, return a JSON object inside a "detections" array with exactly these fields:
    defectType: one of [paint_finish_defect, tile_misalignment, cracked_damaged_tiles,
      joinery_defect, sealant_silicone_defect, incomplete_mep, general_surface_damage,
      ceiling_defect, damp_water_stain, other]
    description: one plain-English sentence for a non-technical site manager describing
      EXACTLY what you can see in that part of the image.
    boundingBox: { "x": <float>, "y": <float>, "width": <float>, "height": <float> }
      -- decimal fractions of image dimensions (0.0-1.0), (0,0) = top-left
    confidence: float 0.0-1.0

Return ONLY valid JSON matching: { "detections": [ ... ] }
Do NOT include any explanation, markdown, or text outside the JSON object."""


# ── Normalisation helpers ─────────────────────────────────────────────────────

def _normalise_coord(value) -> float:
    """Clamp a coordinate value to [0.0, 1.0]. Auto-detects 0–100 percentage scale."""
    try:
        n = float(value)
    except (TypeError, ValueError):
        return 0.0
    # If > 1.5 assume values are expressed as percentages (0–100)
    if n > 1.5:
        n = n / 100.0
    return max(0.0, min(1.0, n))


def _normalise_bbox(bbox: dict) -> dict:
    x = _normalise_coord(bbox.get("x") or bbox.get("left", 0))
    y = _normalise_coord(bbox.get("y") or bbox.get("top", 0))
    w = _normalise_coord(bbox.get("width") or bbox.get("w", 0))
    h = _normalise_coord(bbox.get("height") or bbox.get("h", 0))
    # Clamp so box never overflows the image
    w = min(w, 1.0 - x)
    h = min(h, 1.0 - y)
    return {"x": x, "y": y, "width": w, "height": h}


def _normalise_defect_type(raw: str) -> str:
    normalised = raw.lower().replace(" ", "_")
    return normalised if normalised in VALID_DEFECT_TYPES else "other"


def _parse_detections(raw_list: list) -> list:
    """Validate and reshape the raw detections array from the model."""
    result = []
    for item in raw_list:
        bbox_raw = item.get("boundingBox") or item.get("bounding_box") or {}
        bbox = _normalise_bbox(bbox_raw)

        # Skip zero-size boxes
        if bbox["width"] == 0 or bbox["height"] == 0:
            continue

        raw_type = item.get("defectType") or item.get("defect_type") or "other"
        defect_type = _normalise_defect_type(str(raw_type))

        confidence = max(0.0, min(1.0, float(item.get("confidence", 0.5))))

        result.append({
            "id": str(uuid.uuid4()),
            "aiDefectType": defect_type,
            "finalDefectType": defect_type,
            "description": str(item.get("description") or "").strip(),
            "boundingBox": bbox,
            "confidence": confidence,
            "reviewState": "pending",
        })
    return result


# ── Public API ────────────────────────────────────────────────────────────────

async def detect_defects(image_bytes: bytes, mime_type: str) -> list:
    """
    Send an image to GPT-4o Vision and return a list of structured defect detections.

    Args:
        image_bytes: Raw bytes of the uploaded image.
        mime_type:   'image/jpeg' or 'image/png'.

    Returns:
        List of detection dicts matching the contracts/api.md shape.

    Raises:
        AiTimeoutError: If the OpenAI call exceeds 60 seconds.
        AiParseError:   If the model returns malformed JSON.
    """
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    data_url = f"data:{mime_type};base64,{b64}"

    try:
        response = await asyncio.wait_for(
            client.chat.completions.create(
                model="gpt-4.1",
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": data_url, "detail": "high"},
                            },
                            {
                                "type": "text",
                                "text": "Identify all construction defects in this site photo.",
                            },
                        ],
                    },
                ],
                max_tokens=2048,
            ),
            timeout=60.0,
        )
    except asyncio.TimeoutError as exc:
        raise AiTimeoutError("OpenAI request timed out after 60 s") from exc

    raw_content = response.choices[0].message.content
    try:
        parsed = json.loads(raw_content)
        detections_raw = parsed.get("detections", [])
        if not isinstance(detections_raw, list):
            raise ValueError("'detections' is not a list")
    except (json.JSONDecodeError, ValueError) as exc:
        raise AiParseError(f"Could not parse model response: {exc}") from exc

    return _parse_detections(detections_raw)
