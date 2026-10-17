"""Scene analysis, situation prediction and species first-aid guidance.

Predictions are heuristic estimates from image cues + handler observations.
They are NOT a veterinary diagnosis.
"""
import cv2
import numpy as np

DISASTERS = ["Auto-detect", "Flood / water", "Fire / smoke", "Earthquake / collapse",
             "Road accident", "Extreme heat", "Extreme cold / snow", "Trapped / entangled",
             "Stray / abandoned"]


def analyze_scene(image):
    """Estimate scene conditions from pixel statistics."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    total = float(gray.size)
    brightness = float(v.mean())
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    water = float(((h >= 90) & (h <= 130) & (s > 50) & (v > 50)).sum()) / total
    mud = float(((h >= 8) & (h <= 25) & (s > 60) & (v > 40) & (v < 180)).sum()) / total
    smoke = float(((s < 30) & (v > 90) & (v < 200)).sum()) / total
    fire = float((((h <= 15) | (h >= 170)) & (s > 150) & (v > 180)).sum()) / total
    blood = float((((h <= 6) | (h >= 172)) & (s > 140) & (v > 50) & (v < 160)).sum()) / total
    snow = float(((s < 25) & (v > 215)).sum()) / total
    debris = float(cv2.Canny(gray, 80, 200).mean()) / 255.0

    cues = []
    if brightness < 60: cues.append("Low light / night conditions")
    if blur < 60: cues.append("Image is blurry (motion or poor focus)")
    if water > 0.25: cues.append("Large water area visible (possible flooding)")
    if mud > 0.25: cues.append("Mud / soil dominant (possible landslide or flood debris)")
    if fire > 0.02: cues.append("Bright flame-like colors detected")
    if smoke > 0.45: cues.append("Hazy grey tones (possible smoke or dust)")
    if snow > 0.35: cues.append("Snow / ice-like white areas")
    if debris > 0.18: cues.append("High clutter (possible rubble / debris)")
    if blood > 0.01: cues.append("Dark-red patches that may be blood (verify!)")
    return {
        "brightness": brightness, "sharpness": blur, "water": water, "mud": mud,
        "smoke": smoke, "fire": fire, "blood": blood, "snow": snow, "debris": debris,
        "cues": cues,
    }


def guess_disaster(scene):
    scores = {
        "Flood / water": scene["water"] * 2 + scene["mud"],
        "Fire / smoke": scene["fire"] * 10 + scene["smoke"] * 0.6,
        "Earthquake / collapse": scene["debris"] * 2 + scene["smoke"] * 0.3,
        "Extreme cold / snow": scene["snow"] * 1.5,
    }
    best = max(scores, key=scores.get)
    return best if scores[best] > 0.35 else "Stray / abandoned"


DISASTER_RISKS = {
    "Flood / water": ["Hypothermia", "Near-drowning / water in lungs", "Exhaustion",
                      "Infection from contaminated water", "Skin & paw rot"],
    "Fire / smoke": ["Smoke inhalation (breathing distress)", "Burns on paws, face, ears",
                     "Eye irritation", "Dehydration", "Shock"],
    "Earthquake / collapse": ["Crush injury", "Fractures", "Internal bleeding",
                              "Dehydration from being trapped", "Panic / fear aggression"],
    "Road accident": ["Fractures", "Internal bleeding", "Head trauma", "Shock", "Spinal injury"],
    "Extreme heat": ["Heatstroke", "Dehydration", "Burned paw pads", "Collapse"],
    "Extreme cold / snow": ["Hypothermia", "Frostbite on ears/tail/paws", "Weakness"],
    "Trapped / entangled": ["Cuts / constriction wounds", "Limb swelling", "Dehydration",
                            "Starvation", "Panic injuries"],
    "Stray / abandoned": ["Malnutrition", "Dehydration", "Skin disease / parasites",
                          "Untreated wounds"],
}

DISASTER_ACTIONS = {
    "Flood / water": ["Move the animal to dry, high ground", "Dry with towels and wrap in a dry blanket",
                      "Keep it warm; do NOT use very hot water or hair-dryer on high",
                      "Offer clean drinking water only (not flood water)"],
    "Fire / smoke": ["Move to fresh air immediately", "Cool burns with clean cool (not ice) water for 10 min",
                     "Cover burns loosely with clean damp cloth", "Do not apply butter, oil or creams"],
    "Earthquake / collapse": ["Do not enter unstable structures", "Keep the animal still; suspect fractures",
                              "Slide onto a flat board/blanket to move", "Small sips of water only if alert"],
    "Road accident": ["Secure the area from traffic first", "Keep spine and neck straight while moving",
                      "Apply firm pressure on bleeding with clean cloth", "Keep warm and quiet"],
    "Extreme heat": ["Move to shade", "Cool with room-temperature water on belly & paws",
                     "Offer small amounts of water", "Do NOT use ice water"],
    "Extreme cold / snow": ["Bring to a warm shelter gradually", "Wrap in dry blankets",
                            "Warm water bottles (wrapped) near chest", "Do not rub frostbitten areas"],
    "Trapped / entangled": ["Approach slowly; cover the head with a cloth to calm",
                            "Cut entanglement carefully, away from skin", "Do not pull embedded wire/rope out of wounds"],
    "Stray / abandoned": ["Offer water and small portions of plain food", "Keep it in a quiet, safe place",
                          "Avoid direct contact if it is aggressive"],
}

SPECIES_CARE = {
    "dog": {"handling": "Approach sideways, speak calmly. Use a soft muzzle (cloth) if injured - pain makes dogs bite.",
            "treatments": ["Clean wounds with saline / clean water", "Bandage bleeding firmly",
                           "Splint a broken leg with rolled newspaper + tape", "Oral rehydration (small sips)",
                           "Vet: pain relief, antibiotics, X-ray for fractures"],
            "avoid": "Never give human painkillers (paracetamol, ibuprofen) - toxic to dogs."},
    "cat": {"handling": "Wrap in a towel ('burrito') to prevent scratches. Keep in a closed box.",
            "treatments": ["Keep warm and dark", "Clean wounds with saline", "Small sips of water",
                           "Vet: fluids, antibiotics, oxygen for smoke inhalation"],
            "avoid": "Paracetamol, essential oils and lilies are deadly for cats."},
    "bird": {"handling": "Cover with a light cloth, hold gently around the body (not chest-compressing).",
             "treatments": ["Place in ventilated cardboard box with towel", "Keep warm, dark and quiet",
                            "Do not force water/food", "Wildlife rehabber / avian vet"],
             "avoid": "Do not squeeze the chest - birds cannot breathe if compressed."},
    "horse": {"handling": "Approach from the front-side, use halter. Stay out of kicking range.",
              "treatments": ["Hose leg wounds with cold water", "Pressure bandage on bleeding",
                             "Offer water in small amounts", "Equine vet: colic & fracture assessment"],
              "avoid": "Do not try to make a horse with a suspected fracture walk."},
    "cow": {"handling": "Move slowly, use barriers/gates; avoid getting cornered.",
            "treatments": ["Shade and water", "Clean wounds", "Large-animal vet: fluids, antibiotics"],
            "avoid": "Do not lift a downed cow by the tail or legs."},
    "sheep": {"handling": "Corner gently and hold under the chin and rump.",
              "treatments": ["Dry wet fleece (hypothermia risk)", "Clean wounds, check for maggots",
                             "Vet / livestock officer"],
              "avoid": "Do not flip a pregnant ewe on its back for long."},
    "elephant": {"handling": "DANGEROUS - keep distance. Only forest dept / wildlife team.",
                 "treatments": ["Keep crowds away", "Wildlife vet: sedation & treatment"],
                 "avoid": "Never approach or feed."},
    "bear": {"handling": "DANGEROUS - keep at least 100 m away.",
             "treatments": ["Call wildlife department only"], "avoid": "Never approach, corner or feed."},
    "zebra": {"handling": "Wild animal - keep distance.", "treatments": ["Wildlife vet only"],
              "avoid": "Do not approach from behind."},
    "giraffe": {"handling": "Wild animal - keep distance.", "treatments": ["Wildlife vet only"],
                "avoid": "Do not approach."},
}
DEFAULT_CARE = {"handling": "Approach slowly and calmly; protect yourself with gloves/towel.",
                "treatments": ["Keep warm and quiet", "Clean wounds with clean water",
                               "Offer water if conscious", "Contact a vet / rescue team"],
                "avoid": "Do not give human medicines."}


def predict_condition(animals, scene, disaster, health_screen):
    """Combine detections, scene cues, disaster type and handler checklist."""
    if disaster == "Auto-detect":
        disaster = guess_disaster(scene)
    observations = {o["key"] for o in health_screen.get("observations", [])}

    risk = 1
    if scene["blood"] > 0.01: risk += 2
    if scene["fire"] > 0.02 or scene["water"] > 0.3: risk += 1
    if scene["debris"] > 0.18: risk += 1
    if health_screen.get("status") == "urgent_attention": risk += 3
    elif health_screen.get("status") == "concern_observed": risk += 1
    risk = min(risk, 5)
    level = {1: "LOW", 2: "LOW", 3: "MODERATE", 4: "HIGH", 5: "CRITICAL"}[risk]

    likely = list(DISASTER_RISKS.get(disaster, []))
    if "difficulty_breathing" in observations: likely.insert(0, "Respiratory distress")
    if "heavy_bleeding_or_deep_wound" in observations: likely.insert(0, "Severe blood loss")
    if "unable_to_stand" in observations: likely.insert(0, "Possible fracture / spinal injury / shock")

    species = []
    for label in dict.fromkeys(a["label"] for a in animals):
        species.append({"label": label, **SPECIES_CARE.get(label, DEFAULT_CARE)})

    return {
        "disaster": disaster, "risk_score": risk, "risk_level": level,
        "likely_conditions": likely[:6],
        "before_rescue": ["Make sure YOU are safe first"] + DISASTER_ACTIONS.get(disaster, []) +
                         ["Take clear photos and note the exact location", "Call / alert the rescue team"],
        "species": species or [{"label": "unknown animal", **DEFAULT_CARE}],
    }


def format_prediction(p):
    lines = ["", "SITUATION PREDICTION (heuristic, not a diagnosis)", "-" * 32,
             f"Likely situation: {p['disaster']}",
             f"Estimated risk: {p['risk_level']} ({p['risk_score']}/5)", "Possible conditions:"]
    lines += [f"  - {c}" for c in p["likely_conditions"]]
    lines += ["", "WHAT TO DO BEFORE RESCUE ARRIVES", "-" * 32]
    lines += [f"  {i}. {a}" for i, a in enumerate(p["before_rescue"], 1)]
    for s in p["species"]:
        lines += ["", f"CARE GUIDE: {s['label'].upper()}", "-" * 32, f"Handling: {s['handling']}",
                  "Recommended treatments:"] + [f"  - {t}" for t in s["treatments"]] + [f"Avoid: {s['avoid']}"]
    return "\n".join(lines) + "\n"
