from supabase import create_client, Client
from config import SUPABASE_URL, SUPABASE_SERVICE_KEY

_client: Client | None = None


def get_client() -> Client:
    global _client
    if _client is None:
        if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
            raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set")
        _client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    return _client


def get_teacher_embeddings(teacher_id: str) -> list[list[float]]:
    """Fetch stored face embeddings for a teacher."""
    client = get_client()
    result = (
        client.table("std_teacher_faces")
        .select("face_embeddings")
        .eq("teacher_id", teacher_id)
        .single()
        .execute()
    )
    if result.data:
        return result.data["face_embeddings"] or []
    return []


def save_teacher_enrollment(
    teacher_id: str,
    embeddings: list[list[float]],
    device_fingerprint: str | None = None,
) -> dict:
    """Save face embeddings for a new teacher enrollment."""
    client = get_client()
    result = (
        client.table("std_teacher_faces")
        .upsert(
            {
                "teacher_id": teacher_id,
                "face_embeddings": embeddings,
                "device_fingerprint": device_fingerprint,
            },
            on_conflict="teacher_id",
        )
        .execute()
    )
    return result.data[0] if result.data else {}


def update_teacher_embedding(teacher_id: str, embeddings: list[list[float]]) -> None:
    """Update teacher embeddings (after adaptive blend)."""
    client = get_client()
    client.table("std_teacher_faces").update(
        {"face_embeddings": embeddings}
    ).eq("teacher_id", teacher_id).execute()


def save_teacher_attendance(
    teacher_id: str,
    date: str,
    check_type: str,  # "check_in" or "check_out"
    confidence: float,
    anti_spoof_score: float,
    device_fingerprint: str | None = None,
    service_point_id: str | None = None,
) -> dict:
    """Record teacher check-in or check-out."""
    client = get_client()
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc).isoformat()

    if check_type == "check_in":
        data = {
            "teacher_id": teacher_id,
            "date": date,
            "check_in": now,
            "confidence_in": confidence,
            "anti_spoof_score_in": anti_spoof_score,
            "device_fingerprint": device_fingerprint,
            "service_point_id": service_point_id,
        }
    else:
        data = {
            "teacher_id": teacher_id,
            "date": date,
            "check_out": now,
            "confidence_out": confidence,
            "anti_spoof_score_out": anti_spoof_score,
        }

    result = (
        client.table("std_teacher_attendance")
        .upsert(data, on_conflict="teacher_id,date")
        .execute()
    )
    return result.data[0] if result.data else {}


def get_teacher_attendance_today(teacher_id: str, date: str) -> dict | None:
    """Get teacher's attendance record for a specific date."""
    client = get_client()
    result = (
        client.table("std_teacher_attendance")
        .select("*")
        .eq("teacher_id", teacher_id)
        .eq("date", date)
        .maybe_single()
        .execute()
    )
    return result.data


def get_teacher_face(teacher_id: str) -> dict | None:
    """Get teacher's face enrollment data."""
    client = get_client()
    result = (
        client.table("std_teacher_faces")
        .select("*")
        .eq("teacher_id", teacher_id)
        .maybe_single()
        .execute()
    )
    return result.data
