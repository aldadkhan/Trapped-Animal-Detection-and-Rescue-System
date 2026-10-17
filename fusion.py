def calculate_overlap(box_a, box_b):
    """Calculate Intersection over Union (IoU) for two [x1, y1, x2, y2] boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    intersection_width = max(0, ix2 - ix1)
    intersection_height = max(0, iy2 - iy1)
    intersection_area = intersection_width * intersection_height

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union_area = area_a + area_b - intersection_area

    if union_area == 0:
        return 0.0

    return intersection_area / union_area


def fuse_detections(animal_detections, thermal_hotspots, overlap_threshold=0.10):
    """
    Compare animal detections with thermal hotspots.
    Boxes must already be in the same calibrated image coordinate system.
    """
    if not 0 <= overlap_threshold <= 1:
        raise ValueError("Overlap threshold must be between 0 and 1.")

    fused_results = []

    for animal in animal_detections:
        animal_box = animal["bbox"]
        matching_hotspots = []

        for hotspot in thermal_hotspots:
            overlap = calculate_overlap(animal_box, hotspot["bbox"])
            if overlap >= overlap_threshold:
                matching_hotspots.append({
                    "bbox": hotspot["bbox"],
                    "overlap": overlap
                })

        if matching_hotspots:
            status = (
                "VISIBLE ANIMAL DETECTION WITH OVERLAPPING "
                "THERMAL HOTSPOT (DEMO ONLY)"
            )
        else:
            status = (
                "VISIBLE ANIMAL DETECTION; NO MATCHING HOTSPOT "
                "(DEMO ONLY)"
            )

        fused_results.append({
            "label": animal["label"],
            "confidence": animal["confidence"],
            "bbox": animal_box,
            "status": status,
            "matching_hotspots": matching_hotspots
        })

    return fused_results
