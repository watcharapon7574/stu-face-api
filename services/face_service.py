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
    """Check if the face in the image is real (not a photo/video)."""
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
    """Process verification: anti-spoofing + face matching for all frames."""
    spoofing_results = []
    match_results = []
    frame_embeddings = []

    for frame in frames:
        # Anti-spoofing check
        spoof = check_anti_spoofing(frame)
        spoofing_results.append(spoof)

        if not spoof["is_real"]:
            match_results.append({"matched": False, "confidence": 0.0})
            continue

        # Extract embedding and compare
        try:
            emb = extract_embedding(frame)
            frame_embeddings.append(emb)
            match = compare_embeddings(emb, stored_embeddings)
            match_results.append(match)
        except ValueError:
            match_results.append({"matched": False, "confidence": 0.0})

    # Overall result: majority of frames must pass
    real_count = sum(1 for s in spoofing_results if s["is_real"])
    matched_count = sum(1 for m in match_results if m.get("matched"))
    total = len(frames)

    all_real = real_count == total
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
        "matched": all_real and majority_matched,
        "confidence": round(avg_confidence, 4),
        "is_real": all_real,
        "anti_spoof_score": round(avg_spoof_score, 4),
        "spoofing_scores": [round(s["score"], 4) for s in spoofing_results],
        "frame_results": {
            "total": total,
            "real": real_count,
            "matched": matched_count,
        },
        "best_embedding": best_embedding,
    }
