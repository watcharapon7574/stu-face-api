from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from services.face_service import decode_base64_image, process_enrollment
from services.supabase_client import save_teacher_enrollment, get_teacher_face
from config import MIN_FRAMES

router = APIRouter()


class EnrollRequest(BaseModel):
    teacher_id: str
    images: list[str]  # base64 encoded images (3 photos)
    device_fingerprint: str | None = None
    skip_anti_spoof: bool = False


class EnrollResponse(BaseModel):
    success: bool
    embeddings_count: int
    spoofing_results: list[dict]
    message: str


@router.post("/enroll", response_model=EnrollResponse)
async def enroll_teacher(req: EnrollRequest):
    """Enroll a teacher's face with 3 photos from different angles."""

    if len(req.images) < MIN_FRAMES:
        raise HTTPException(
            status_code=400,
            detail=f"ต้องส่งรูปอย่างน้อย {MIN_FRAMES} รูป (ได้ {len(req.images)})",
        )

    # Decode images
    try:
        images = [decode_base64_image(img) for img in req.images]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"รูปภาพไม่ถูกต้อง: {str(e)}")

    # Process enrollment: extract embeddings + anti-spoofing
    result = process_enrollment(images, skip_anti_spoof=req.skip_anti_spoof)

    if not req.skip_anti_spoof and not result["all_real"]:
        return EnrollResponse(
            success=False,
            embeddings_count=0,
            spoofing_results=result["spoofing_results"],
            message="ตรวจพบว่าภาพไม่ใช่ใบหน้าจริง กรุณาถ่ายรูปใหม่",
        )

    if result["valid_count"] < MIN_FRAMES:
        return EnrollResponse(
            success=False,
            embeddings_count=result["valid_count"],
            spoofing_results=result["spoofing_results"],
            message=f"ตรวจจับใบหน้าได้แค่ {result['valid_count']}/{len(req.images)} รูป กรุณาถ่ายใหม่",
        )

    # Save to database
    save_teacher_enrollment(
        teacher_id=req.teacher_id,
        embeddings=result["embeddings"],
        device_fingerprint=req.device_fingerprint,
    )

    return EnrollResponse(
        success=True,
        embeddings_count=result["valid_count"],
        spoofing_results=result["spoofing_results"],
        message="ลงทะเบียนใบหน้าสำเร็จ",
    )
