import os

import requests

key = os.environ.get("FIRMS_MAP_KEY", "")
print("key length:", len(key))

# Historical (dated) request — a known big-fire window
url_hist = (
    f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{key}"
    f"/VIIRS_SNPP_SP/-125,31,-102,49/10/2018-11-08"
)
r = requests.get(url_hist, timeout=120)
print("\n[historical] status:", r.status_code)
print("[historical] body:", r.text[:500])

# Recent (no date) request
url_recent = (
    f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{key}"
    f"/VIIRS_SNPP_NRT/-125,31,-102,49/2"
)
r2 = requests.get(url_recent, timeout=120)
print("\n[recent] status:", r2.status_code)
print("[recent] body:", r2.text[:500])
