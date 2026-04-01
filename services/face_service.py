import base64
import io
import numpy as np
from PIL import Image
from deepface import DeepFace
from config import (
    FACE_MODEL,
    FACE_DETECTOR,
    DISTANCE_METRIC,
    FACE_MATCH_THRESHOLD,
    ANTI_SPOOF_ENABLED,
    BLEND_OLD_RATIO,
    BLEND_NEW_RATIO,
)


def decode_base64_image(base64_str: str) -> np.ndarray:
    """Decode base64 image string to numpy array (RGB)."""
    # Remove data URL prefix if present
    if "," in base64_str:
        base64_str = base64_str.split(",", 1)[1]

    img_bytes = base64.b64decode(base64_str)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return np.array(img)


def extract_embedding(image: np.ndarray) -> list[float]:
    """Extract face embedding from an image using DeepFace."""
    result = DeepFace.represent(
        img_path=image,
        model_name=FACE_MODEL,
        detector_backend=FACE_DETECTOR,
        enforce_detection=True,
    )
    if not result:
        raise ValueError("No face detected in image")
    return result[0]["embedding"]


def check_anti_spoofing(image: np.ndarray) -> dict:
    """Check if the face in the image is real (not a photo/video) using MiniFASNet."""
    if not ANTI_SPOOF_ENABLED:
        return {"is_real": True, "score": 1.0}

    try:
        result = DeepFace.extract_faces(
            img_path=image,
            detector_backend=FACE_DETECTOR,
            anti_spoofing=True,
        )
        if not result:
            return {"is_real": False, "score": 0.0}

        face = result[0]
        is_real = face.get("is_real", False)
        score = face.get("antispoof_score", 0.0)
        return {"is_real": is_real, "score": score}
    except Exception:
        return {"is_real": False, "score": 0.0}


def check_frame_variance(frames: list[np.ndarray], threshold: float = 2.0) -> dict:
    """Check liveness by comparing pixel differences between frames.

    Real faces have micro-movements between frames (blinks, breathing, tiny shifts).
    Photos/static images are nearly identical across frames.

    Args:
        frames: list of RGB images as numpy arrays
        threshold: minimum mean pixel difference to consider "alive"

    Returns:
        is_real: True if frames show enough variance (real face)
        score: normalized variance score (0-1)
        mean_diff: average pixel difference between consecutive frames
    """
    if len(frames) < 2:
        return {"is_real": True, "score": 1.0, "mean_diff": 999.0}

    diffs = []
    for i in range(len(frames) - 1):
        # Convert to grayscale for comparison
        gray1 = np.mean(frames[i], axis=2)
        gray2 = np.mean(frames[i + 1], axis=2)

        # Resize to same size if needed
        if gray1.shape != gray2.shape:
            min_h = min(gray1.shape[0], gray2.shape[0])
            min_w = min(gray1.shape[1], gray2.shape[1])
            gray1 = gray1[:min_h, :min_w]
            gray2 = gray2[:min_h, :min_w]

        diff = np.abs(gray1.astype(float) - gray2.astype(float))
        diffs.append(np.mean(diff))

    mean_diff = float(np.mean(diffs))
    # Normalize: score 0-1 where 1 = very different (definitely real)
    score = min(mean_diff / 10.0, 1.0)
    is_real = mean_diff >= threshold

    return {"is_real": is_real, "score": round(score, 4), "mean_diff": round(mean_diff, 2)}


def compare_embeddings(
    live_embedding: list[float], stored_embeddings: list[list[float]]
) -> dict:
    """Compare a live embedding against stored embeddings. Returns best match."""
    if not stored_embeddings:
        return {"matched": False, "confidence": 0.0, "best_index": -1}

    live = np.array(live_embedding)
    best_similarity = 0.0
    best_index = -1

    for i, stored in enumerate(stored_embeddings):
        stored_vec = np.array(stored)

        if DISTANCE_METRIC == "cosine":
            # Cosine similarity (1 - cosine distance)
            dot = np.dot(live, stored_vec)
            norm = np.linalg.norm(live) * np.linalg.norm(stored_vec)
            similarity = dot / norm if norm > 0 else 0.0
        else:
            # Euclidean distance → convert to similarity
            dist = np.linalg.norm(live - stored_vec)
            similarity = 1.0 / (1.0 + dist)

        if similarity > best_similarity:
            best_similarity = similarity
            best_index = i

    # For cosine: threshold is distance, so similarity threshold = 1 - FACE_MATCH_THRESHOLD
    matched = best_similarity >= (1.0 - FACE_MATCH_THRESHOLD)

    return {
        "matched": matched,
        "confidence": round(best_similarity, 4),
        "best_index": best_index,
    }


def blend_embeddings(
    stored: list[float], live: list[float]
) -> list[float]:
    """Adaptive blend: 80% stored + 20% live."""
    stored_arr = np.array(stored)
    live_arr = np.array(live)
    blended = stored_arr * BLEND_OLD_RATIO + live_arr * BLEND_NEW_RATIO
    # Normalize
    norm = np.linalg.norm(blended)
    if norm > 0:
        blended = blended / norm
    return blended.tolist()


def process_enrollment(images: list[np.ndarray], skip_anti_spoof: bool = False) -> dict:
    """Process enrollment: extract embeddings + optional anti-spoofing check."""
    embeddings = []
    spoofing_results = []

    for img in images:
        if skip_anti_spoof:
            spoof = {"is_real": True, "score": 1.0}
        else:
            spoof = check_anti_spoofing(img)
        spoofing_results.append(spoof)

        if not spoof["is_real"]:
            continue

        # Extract embedding
        try:
            emb = extract_embedding(img)
            embeddings.append(emb)
        except ValueError:
            continue

    all_real = all(s["is_real"] for s in spoofing_results)

    return {
        "embeddings": embeddings,
        "spoofing_results": spoofing_results,
        "all_real": all_real,
        "valid_count": len(embeddings),
    }


def process_verification(
    frames: list[np.ndarray], stored_embeddings: list[list[float]]
) -> dict:
    """Process verification: frame variance liveness + face matching for all frames."""
    spoofing_results = []
    match_results = []
    frame_embeddings = []

    # Primary anti-spoofing: multi-frame variance check
    # Real face = micro-movements between frames, photo = identical frames
    variance = check_frame_variance(frames)

    for frame in frames:
        # MiniFASNet check (soft — record score only, don't block)
        spoof = check_anti_spoofing(frame)
        spoofing_results.append(spoof)

        # Always try face matching
        try:
            emb = extract_embedding(frame)
            frame_embeddings.append(emb)
            match = compare_embeddings(emb, stored_embeddings)
            match_results.append(match)
        except ValueError:
            match_results.append({"matched": False, "confidence": 0.0})

    # Overall result
    real_count = sum(1 for s in spoofing_results if s["is_real"])
    matched_count = sum(1 for m in match_results if m.get("matched"))
    total = len(frames)

    # Liveness: use frame variance (primary) — real face has micro-movements
    is_real = variance["is_real"]
    majority_matched = matched_count > total / 2

    # Average confidence from matched frames
    matched_confidences = [
        m["confidence"] for m in match_results if m.get("matched")
    ]
    avg_confidence = (
        sum(matched_confidences) / len(matched_confidences)
        if matched_confidences
        else 0.0
    )

    # Average anti-spoof score
    avg_spoof_score = (
        sum(s["score"] for s in spoofing_results) / len(spoofing_results)
        if spoofing_results
        else 0.0
    )

    # Best embedding for adaptive update (highest confidence match)
    best_embedding = None
    if frame_embeddings and match_results:
        best_idx = max(
            range(len(match_results)),
            key=lambda i: match_results[i].get("confidence", 0),
        )
        if best_idx < len(frame_embeddings):
            best_embedding = frame_embeddings[best_idx]

    return {
        "matched": is_real and majority_matched,  # must pass liveness + face match
        "confidence": round(avg_confidence, 4),
        "is_real": is_real,
        "anti_spoof_score": round(avg_spoof_score, 4),
        "spoofing_scores": [round(s["score"], 4) for s in spoofing_results],
        "frame_variance": {
            "mean_diff": variance["mean_diff"],
            "score": variance["score"],
            "is_real": variance["is_real"],
        },
        "frame_results": {
            "total": total,
            "real": real_count,
            "matched": matched_count,
        },
        "best_embedding": best_embedding,
    }
