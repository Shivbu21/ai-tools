"""demo.py — CLI demo: 4 scripted job requests, ranked matches with breakdowns.

Also asserts determinism: every request is ranked TWICE and the two runs must
produce identical orderings and identical totals.
Run:  python3 demo.py
"""
from matcher import rank, explain_text

REQUESTS = [
    {"service": "electrician", "area": "Noida Sec 62", "min_rating": 0.0,
     "label": "1. Electrician needed in Noida Sec 62"},
    {"service": "plumber", "area": "Greater Noida", "min_rating": 4.0,
     "label": "2. Plumber in Greater Noida, min 4.0 stars"},
    {"service": "carpenter", "area": "Ghaziabad", "min_rating": 0.0,
     "label": "3. Carpenter in Ghaziabad"},
    {"service": "electrician", "area": "Gurgaon", "min_rating": 4.5,
     "label": "4. Top-rated electrician in Gurgaon (min 4.5)"},
]


def fingerprint(results):
    return [(s["pro"]["id"], s["total"]) for s in results]


def main():
    for req in REQUESTS:
        print("=" * 72)
        print(req["label"],
              f"[service={req['service']}, area={req['area']}, min_rating={req['min_rating']}]")
        print("=" * 72)
        run1 = rank(req["service"], req["area"], req["min_rating"])
        run2 = rank(req["service"], req["area"], req["min_rating"])
        # determinism assert: same input -> same ranking, run twice
        assert fingerprint(run1) == fingerprint(run2), \
            f"NON-DETERMINISTIC ranking for {req['label']}!"
        if not run1:
            print("  no matching pros.\n")
            continue
        for s in run1[:3]:
            print(explain_text(s))
            print()
        print(f"  determinism check: 2 runs identical "
              f"({len(run1)} candidates) ✓\n")
    print("ALL DETERMINISM CHECKS PASSED.")


if __name__ == "__main__":
    main()
