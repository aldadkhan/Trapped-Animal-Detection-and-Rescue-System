HEALTH_STATUS_LABELS = {
    "not_assessed": "Not assessed",
    "urgent_attention": "Urgent signs reported - contact a veterinarian or qualified rescuer",
    "concern_observed": "Concern reported - veterinary/rescue assessment recommended",
    "no_concern_reported": "No listed concern reported - this is not medical clearance",
}


def format_analysis_report(
    *,
    created_at,
    image_shape,
    model,
    confidence_threshold,
    animals,
    thermal_shape,
    hotspots,
    hotspot_percentile,
    fusion_enabled,
    fusion_results,
    health_screen,
    source="Uploaded or captured visible-light image",
):
    height, width = image_shape
    lines = [
        "FUR-REAL ANIMAL SEARCH ANALYSIS",
        "=" * 32,
        f"Created: {created_at}",
        f"Input: {source}",
        "",
        "VISIBLE-LIGHT ANIMAL DETECTION",
        "-" * 32,
        f"Image size: {width} x {height} pixels",
        f"Detection model: {model}",
        f"Minimum confidence: {confidence_threshold:.0%}",
        "Supported animal labels: bird, cat, dog, horse, sheep, cow, "
        "elephant, bear, zebra, giraffe.",
    ]

    if animals:
        lines.append(f"Result: {len(animals)} supported animal candidate(s) detected.")
        for index, animal in enumerate(animals, start=1):
            lines.append(
                f"  {index}. {animal['label']} - model score "
                f"{animal['confidence']:.0%}; box {animal['bbox']}"
            )
    else:
        lines.extend(
            [
                "Result: No supported animal candidates were detected.",
                "Meaning: This does not prove that no animal is present. "
                "Blur, low light, obstruction, distance, or unsupported "
                "species can cause missed detections.",
            ]
        )

    lines.extend(["", "THERMAL IMAGE REVIEW", "-" * 32])
    if thermal_shape is None:
        lines.append("Result: No thermal-camera image was provided.")
    else:
        thermal_height, thermal_width = thermal_shape
        lines.append(f"Image size: {thermal_width} x {thermal_height} pixels")
        lines.append(
            f"Method: Relative image intensity at the {hotspot_percentile}th "
            "percentile; not radiometric temperature measurement."
        )
        if hotspots:
            lines.append(
                f"Result: {len(hotspots)} relative bright-region candidate(s)."
            )
            for index, hotspot in enumerate(hotspots, start=1):
                lines.append(
                    f"  {index}. box {hotspot['bbox']}; "
                    f"area {hotspot['area']:.0f} pixels"
                )
        else:
            lines.append("Result: No regions met the selected hotspot threshold.")
        lines.append(
            "These regions are not proof of an animal, body heat, or a "
            "specific temperature."
        )

    lines.extend(["", "SENSOR-FUSION COMPARISON", "-" * 32])
    if not fusion_enabled:
        lines.append(
            "Not performed. Fusion requires calibrated and registered images "
            "in the same coordinate system."
        )
    elif not animals:
        lines.append("Performed; no visible animal detections were available to compare.")
    else:
        for index, result in enumerate(fusion_results, start=1):
            match_count = len(result["matching_hotspots"])
            lines.append(
                f"  {index}. {result['label']}: {match_count} overlapping "
                "thermal candidate(s)."
            )

    health_status = HEALTH_STATUS_LABELS.get(
        health_screen["status"],
        "Unrecognized health-observation status",
    )
    lines.extend(
        [
            "",
            "HANDLER-REPORTED HEALTH OBSERVATIONS",
            "-" * 32,
            f"Status: {health_status}",
            "This checklist is not an AI medical assessment or diagnosis.",
        ]
    )
    if health_screen["observations"]:
        for observation in health_screen["observations"]:
            lines.append(f"  - {observation['label']}")
    elif health_screen["status"] == "no_concern_reported":
        lines.append("  - No listed concerns reported on the checklist.")
    else:
        lines.append("  - Checklist not completed.")

    lines.extend(
        [
            "",
            "IMPORTANT LIMITATIONS",
            "-" * 32,
            "The animal model detects supported visible animal classes; it "
            "does not diagnose injury or illness.",
            "A simulated thermal-style view made from visible pixels is not "
            "thermal data and is not used in this analysis.",
            "Every candidate requires safe human verification. Do not use "
            "this prototype as the sole basis for rescue or treatment decisions.",
        ]
    )
    return "\n".join(lines) + "\n"
