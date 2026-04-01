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
    try:
        client = get_client()
        result = (
            client.table("std_teacher_faces")
            .select("face_embeddings")
            .eq("teacher_id", teacher_id)
            .maybe_single()
            .execute()
        )
        if result.data:
            return result.data["face_embeddings"] or []
    except Exception:
        pass
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


def _get_check_in_time(service_point_id: str | None) -> str:
    """Get check-in time with grace period for service units.

    Service units (non-headquarters) get +30 min grace period.
    During grace period (e.g. 9:01-9:30), record a random time
    between 30 min before deadline (e.g. 8:30-9:00).
    """
    from datetime import datetime, timezone, timedelta
    import random

    tz = timezone(timedelta(hours=7))  # Thailand
    now = datetime.now(tz)

    if not service_point_id:
        return now.astimezone(timezone.utc).isoformat()

    try:
        client = get_client()

        # Check if headquarters
        sp_result = (
            client.table("std_service_points")
            .select("is_headquarters")
            .eq("id", service_point_id)
            .maybe_single()
            .execute()
        )

        is_hq = sp_result.data.get("is_headquarters", False) if sp_result.data else False

        if is_hq:
            return now.astimezone(timezone.utc).isoformat()

        # Get check_in_end from settings
        settings_result = (
            client.table("std_teacher_settings")
            .select("value")
            .eq("key", "check_in_end")
            .maybe_single()
            .execute()
        )

        check_in_end_str = "09:00"
        if settings_result.data:
            check_in_end_str = settings_result.data["value"]

        # Parse check_in_end
        h, m = map(int, check_in_end_str.split(":"))
        deadline = now.replace(hour=h, minute=m, second=0, microsecond=0)
        grace_end = deadline + timedelta(minutes=30)
        random_start = deadline - timedelta(minutes=30)

        # If within grace period (after deadline but before grace_end)
        if deadline < now <= grace_end:
            # Random time between (deadline - 30min) and deadline
            random_seconds = random.randint(0, 30 * 60)
            fake_time = random_start + timedelta(seconds=random_seconds)
            return fake_time.astimezone(timezone.utc).isoformat()

    except Exception:
        pass

    return now.astimezone(timezone.utc).isoformat()


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

    if check_type == "check_in":
        # Apply grace period for service units
        check_in_time = _get_check_in_time(service_point_id)
        data = {
            "teacher_id": teacher_id,
            "date": date,
            "check_in": check_in_time,
            "confidence_in": confidence,
            "anti_spoof_score_in": anti_spoof_score,
            "device_fingerprint": device_fingerprint,
            "service_point_id": service_point_id,
        }
    else:
        now = datetime.now(timezone.utc).isoformat()
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
