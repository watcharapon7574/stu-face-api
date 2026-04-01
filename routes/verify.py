from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from services.face_service import (
    decode_base64_image,
    process_verification,
    blend_embeddings,
)
from services.supabase_client import (
    get_teacher_embeddings,
    update_teacher_embedding,
    save_teacher_attendance,
)
from config import MIN_FRAMES, MAX_FRAMES

router = APIRouter()


class VerifyRequest(BaseModel):
    teacher_id: str
    frames: list[str]  # base64 encoded frames (3-5 auto-captured)
    device_fingerprint: str | None = None
    service_point_id: str | None = None
    check_type: str = "check_in"  # "check_in" or "check_out"
    date: str  # YYYY-MM-DD


class VerifyResponse(BaseModel):
    matched: bool
    confidence: float
    is_real: bool
    anti_spoof_score: float
    spoofing_scores: list[float]
    frame_results: dict
    attendance_saved: bool
    message: str


@router.post("/verify", response_model=VerifyResponse)
async def verify_teacher(req: VerifyRequest):
    """Verify teacher's face with multi-frame anti-spoofing."""

    if len(req.frames) < MIN_FRAMES:
        raise HTTPException(
            status_code=400,
            detail=f"ต้องส่งอย่างน้อย {MIN_FRAMES} เฟรม (ได้ {len(req.frames)})",
        )

    if len(req.frames) > MAX_FRAMES:
        req.frames = req.frames[:MAX_FRAMES]

    # Get stored embeddings
    stored_embeddings = get_teacher_embeddings(req.teacher_id)
    if not stored_embeddings:
        raise HTTPException(
            status_code=404,
            detail="ไม่พบข้อมูลใบหน้า กรุณาลงทะเบียนก่อน",
        )

    # Decode frames
    try:
        frames = [decode_base64_image(f) for f in req.frames]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"รูปภาพไม่ถูกต้อง: {str(e)}")

    # Process verification
    result = process_verification(frames, stored_embeddings)

    attendance_saved = False

    if result["matched"]:
        # Save attendance
        try:
            save_teacher_attendance(
                teacher_id=req.teacher_id,
                date=req.date,
                check_type=req.check_type,
                confidence=result["confidence"],
                anti_spoof_score=result["anti_spoof_score"],
                device_fingerprint=req.device_fingerprint,
                service_point_id=req.service_point_id,
            )
            attendance_saved = True
        except Exception:
            attendance_saved = False

        # Adaptive embedding update (fire-and-forget)
        if result.get("best_embedding") and stored_embeddings:
            try:
                # Find best matching stored embedding index
                best_idx = 0  # default to first
                best_match = result.get("frame_results", {})
                if best_match:
                    best_idx = min(best_idx, len(stored_embeddings) - 1)

                blended = blend_embeddings(
                    stored_embeddings[best_idx], result["best_embedding"]
                )
                stored_embeddings[best_idx] = blended
                update_teacher_embedding(req.teacher_id, stored_embeddings)
            except Exception:
                pass  # Non-critical, don't fail the request

    if not result["is_real"]:
        message = "ตรวจพบว่าภาพไม่ใช่ใบหน้าจริง"
    elif not result["matched"]:
        message = "ใบหน้าไม่ตรงกับข้อมูลที่ลงทะเบียน"
    elif attendance_saved:
        label = "เข้างาน" if req.check_type == "check_in" else "ออกงาน"
        message = f"สแกน{label}สำเร็จ"
    else:
        message = "ยืนยันใบหน้าสำเร็จแต่บันทึกเวลาไม่ได้"

    return VerifyResponse(
        matched=result["matched"],
        confidence=result["confidence"],
        is_real=result["is_real"],
        anti_spoof_score=result["anti_spoof_score"],
        spoofing_scores=result["spoofing_scores"],
        frame_results=result["frame_results"],
        attendance_saved=attendance_saved,
        message=message,
    )


class UpdateEmbeddingRequest(BaseModel):
    teacher_id: str
    frame: str  # single base64 frame


@router.post("/update-embedding")
async def update_embedding(req: UpdateEmbeddingRequest):
    """Adaptive update: blend a new frame's embedding with stored (80/20)."""
    from services.face_service import decode_base64_image, extract_embedding

    stored_embeddings = get_teacher_embeddings(req.teacher_id)
    if not stored_embeddings:
        raise HTTPException(status_code=404, detail="ไม่พบข้อมูลใบหน้า")

    try:
        image = decode_base64_image(req.frame)
        new_embedding = extract_embedding(image)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"ไม่สามารถตรวจจับใบหน้า: {str(e)}")

    # Blend with first stored embedding
    blended = blend_embeddings(stored_embeddings[0], new_embedding)
    stored_embeddings[0] = blended
    update_teacher_embedding(req.teacher_id, stored_embeddings)

    return {"success": True, "message": "อัพเดทข้อมูลใบหน้าสำเร็จ"}
