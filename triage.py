HEALTH_OBSERVATIONS = (
    ("heavy_bleeding_or_deep_wound", "Heavy bleeding or a deep/open wound"),
    ("difficulty_breathing", "Difficulty breathing"),
    ("unable_to_stand", "Unable to stand or move"),
    ("unresponsive_or_very_weak", "Unresponsive or severely weak"),
    ("eye_or_face_injury", "Visible eye or face injury"),
    ("swelling_or_limping", "Unusual swelling or limping"),
)

URGENT_OBSERVATIONS = {
    "heavy_bleeding_or_deep_wound",
    "difficulty_breathing",
    "unable_to_stand",
    "unresponsive_or_very_weak",
}


def build_health_screen(observations):
    """Summarize user-reported visual observations; this does not diagnose."""
    known_observations = {key for key, _ in HEALTH_OBSERVATIONS}
    unknown_observations = set(observations) - known_observations
    if unknown_observations:
        raise ValueError(
            f"Unknown health observation(s): {', '.join(sorted(unknown_observations))}"
        )

    if observations & URGENT_OBSERVATIONS:
        status = "urgent_attention"
    elif observations:
        status = "concern_observed"
    else:
        status = "no_concern_reported"

    return {
        "status": status,
        "observations": [
            {"key": key, "label": label}
            for key, label in HEALTH_OBSERVATIONS
            if key in observations
        ],
        "source": "handler_reported_not_ai_diagnosis",
    }
