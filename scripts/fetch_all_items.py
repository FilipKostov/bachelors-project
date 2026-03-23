#note: set steam token as variable
import requests
import time

appid=730
url="https://steamcommunity.com/market/search/render/"
out="F:\\steam_market_project\\data\\all_items.txt"
sleep_timer=2
count=100

items=list()
seen=set()
start=0
total=None

while True:
    params={"query": "", "appid": appid, "norender": 1, "sort_column": "name", "sort_dir": "asc", "start": start, "count": count }

    r=requests.get(url, params=params, timeout=30)

    try:
        r.raise_for_status()
    except requests.exceptions.RequestException as e:
        if r.status_code==429:
            time.sleep(60)
            continue
        break

    data=r.json()

    if total is None:
        if "total_count" not in data or not isinstance(data["total_count"], int):
            break
        total=data["total_count"]

    results=data.get("results", [])
    if len(results)==0:
        time.sleep(15)
        continue

    for res in results:
        name=res.get("hash_name") or res.get("market_hash_name") or res.get("name")
        if name and name not in seen:
            seen.add(name)
            items.append(name)

    start+=count
    print(f"{len(items)}/{total}")
    time.sleep(sleep_timer)

    if total and len(items)>=total:
        break

if total is not None and len(seen)!=total:
    print(f"\nScraped {len(seen)} items, but total is {total}!")

f=open(out, "w", encoding="utf-8")
for item in items:
    f.write(item + "\n")
f.close()