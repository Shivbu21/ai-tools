"""pro-matcher — deterministic, explainable pro ranking for home services.

Scoring formula (total out of 100). Service match is a GATE: a pro who does not
offer the requested service scores 0 and is excluded from the ranking entirely.

    area_proximity  (30 pts):  same area = 30 | nearby area = 18 | far/unknown = 6
    rating          (25 pts):  (rating / 5) * 25
    completion_rate (20 pts):  completion_rate * 20        (0..1 in pros.json)
    availability    (25 pts):  (1 - current_workload / 5) * 25   (fewer active jobs = better)

Tie-break (deterministic): higher total first, then higher rating, then more
jobs_completed, then name ascending. Same input -> same ranking, always.
No randomness, no network, no ML model — just arithmetic you can audit.
"""
from __future__ import annotations

import json
import os

BASE = os.path.dirname(os.path.abspath(__file__))

# Nearby-area map. Symmetric-ish; an area not listed here (or not listed as a
# neighbour) counts as "far". Tune to your city.
NEARBY = {
    "Noida Sec 62": ["Noida Sec 18", "Noida Sec 137", "Ghaziabad"],
    "Noida Sec 18": ["Noida Sec 62", "Delhi South"],
    "Noida Sec 137": ["Noida Sec 62", "Greater Noida"],
    "Greater Noida": ["Noida Sec 137", "Noida Sec 62"],
    "Ghaziabad": ["Noida Sec 62", "Delhi East"],
    "Delhi South": ["Noida Sec 18", "Delhi East", "Gurgaon"],
    "Delhi East": ["Ghaziabad", "Delhi South"],
    "Gurgaon": ["Delhi South"],
}

WEIGHTS = {"area_proximity": 30.0, "rating": 25.0, "completion_rate": 20.0,
           "availability": 25.0}


def load_pros(path: str | None = None) -> list[dict]:
    path = path or os.path.join(BASE, "pros.json")
    with open(path) as f:
        data = json.load(f)
    return data["pros"]


def all_areas(pros: list[dict] | None = None) -> list[str]:
    pros = pros if pros is not None else load_pros()
    areas: set[str] = set(NEARBY)
    for p in pros:
        areas.update(p.get("areas", []))
    return sorted(areas)


def all_services(pros: list[dict] | None = None) -> list[str]:
    pros = pros if pros is not None else load_pros()
    svcs: set[str] = set()
    for p in pros:
        svcs.update(s.lower() for s in p.get("services", []))
    return sorted(svcs)


def area_points(pro_area: str, pro_areas: list[str]) -> tuple[float, str]:
    """30 / 18 / 6 with a human-readable label."""
    if pro_area in pro_areas:
        return WEIGHTS["area_proximity"], f"same area ({pro_area})"
    for a in pro_areas:
        if pro_area in NEARBY.get(a, []) or a in NEARBY.get(pro_area, []):
            return 18.0, f"nearby ({a} ~ {pro_area})"
    return 6.0, "far / outside service cluster"


def score_pro(pro: dict, service: str, area: str) -> dict | None:
    """Score one pro. Returns None if the pro fails the service gate."""
    if service.lower() not in [s.lower() for s in pro.get("services", [])]:
        return None  # gate: no service match -> excluded, not zero-ranked

    rating = float(pro.get("rating", 0))
    comp_rate = float(pro.get("completion_rate", 0))
    workload = int(pro.get("current_workload", 0))

    a_pts, a_label = area_points(area, pro.get("areas", []))
    r_pts = (max(0.0, min(5.0, rating)) / 5.0) * WEIGHTS["rating"]
    c_pts = max(0.0, min(1.0, comp_rate)) * WEIGHTS["completion_rate"]
    v_pts = (1.0 - max(0, min(5, workload)) / 5.0) * WEIGHTS["availability"]

    breakdown = [
        ("area_proximity", round(a_pts, 2), a_label),
        ("rating", round(r_pts, 2), f"{rating}/5 stars"),
        ("completion_rate", round(c_pts, 2), f"{comp_rate:.0%} jobs completed successfully"),
        ("availability", round(v_pts, 2),
         f"{workload}/5 active jobs" + (" — fully booked" if workload >= 5 else "")),
    ]
    total = round(sum(p for _, p, _ in breakdown), 2)
    return {"pro": pro, "total": total, "breakdown": breakdown}


def rank(service: str, area: str, min_rating: float = 0.0,
         pros: list[dict] | None = None) -> list[dict]:
    """Ranked matches for a job request. Deterministic: same input, same output."""
    pros = pros if pros is not None else load_pros()
    out = []
    for pro in pros:
        if float(pro.get("rating", 0)) < min_rating:
            continue
        scored = score_pro(pro, service, area)
        if scored is not None:
            out.append(scored)
    # deterministic tie-break: total desc, rating desc, jobs desc, name asc
    out.sort(key=lambda s: (-s["total"],
                            -float(s["pro"].get("rating", 0)),
                            -int(s["pro"].get("jobs_completed", 0)),
                            s["pro"].get("name", "")))
    return out


def explain_text(scored: dict) -> str:
    p = scored["pro"]
    lines = [f"{p['name']} ({p['id']}) — total {scored['total']:.1f}/100"]
    for name, pts, label in scored["breakdown"]:
        maxw = WEIGHTS[name]
        bar = "#" * int(20 * pts / maxw) + "-" * (20 - int(20 * pts / maxw))
        lines.append(f"  {name:15s} [{bar}] {pts:5.1f}/{maxw:.0f}  {label}")
    lines.append(f"  areas: {', '.join(p['areas'])} | jobs done: {p['jobs_completed']} | "
                 f"phone: {p['phone']}")
    return "\n".join(lines)
