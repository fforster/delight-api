#!/usr/bin/env python3
"""
Smoke test for the DELIGHT API.

Uses coordinates from the DELIGHT test dataset to verify the pipeline
runs end-to-end.  Requires the API to be running on localhost:8000.

Usage:
    python test_api.py [API_KEY]
"""

import sys
import requests

API_URL = "http://localhost:8000"
API_KEY = sys.argv[1] if len(sys.argv) > 1 else "changeme"
HEADERS = {"X-API-Key": API_KEY}

# Test objects from DELIGHT's data/testcoords.csv
TEST_CASES = [
    {"oid": "SN2022yrt", "ra": 132.1190120, "dec": 1.3972640},
    {"oid": "SN2004aq", "ra": 186.812042, "dec": 12.749750},
    {"oid": "SN2001ay", "ra": 224.570917, "dec": 6.948889},
]


def test_health():
    r = requests.get(f"{API_URL}/health")
    r.raise_for_status()
    assert r.json()["status"] == "ok", "Health check failed"
    print("[PASS] /health")


def test_auth():
    r = requests.post(
        f"{API_URL}/predict",
        json=TEST_CASES[0],
        headers={"X-API-Key": "wrong-key"},
    )
    assert r.status_code == 401, f"Expected 401, got {r.status_code}"
    print("[PASS] Authentication rejects bad key")


def test_predict(case):
    print(f"\n[TEST] Predicting host for {case['oid']} "
          f"({case['ra']:.4f}, {case['dec']:.4f}) ...")

    r = requests.post(f"{API_URL}/predict", json=case, headers=HEADERS)
    r.raise_for_status()
    result = r.json()

    assert result["oid"] == case["oid"]
    assert len(result["host_predictions"]) == 8

    print(f"  Mean host: RA={result['ra_mean']:.6f}, "
          f"Dec={result['dec_mean']:.6f}")
    print(f"  Scatter: {result['std_pixels']:.3f} pixels")
    for i, p in enumerate(result["host_predictions"]):
        print(f"    [{i}] RA={p['ra']:.6f}  Dec={p['dec']:.6f}")

    print(f"[PASS] {case['oid']}")


if __name__ == "__main__":
    test_health()
    test_auth()
    for case in TEST_CASES:
        test_predict(case)
    print("\nAll tests passed.")
