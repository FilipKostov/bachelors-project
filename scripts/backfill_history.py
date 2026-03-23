#note: reset checkpoint i stavi login token vo variable
import os
import time
from pathlib import Path
import requests
import psycopg2
import psycopg2.extras
from dateutil import parser as dateparser
from datetime import datetime


db_conn="dbname=steam user=postgres password=password host=127.0.0.1 port=5432"
fai_out=Path("F:/steam_market_project/data/all_items.txt")
appid=730
steam_url="https://steamcommunity.com/market/pricehistory/"
agent="Mozilla/5.0 (compatible; ThesisBackfill/1.0; +https://your.site)"
sleep_between_items=0.5
batch_size=500
max_retries=5
curr="USD"
checkpoint_path=Path("F:/steam_market_project/data/backfill_checkpoint.txt")

def load_checkpoint(default=0):
    try:
        pom=int(checkpoint_path.read_text(encoding="utf-8").strip())
        return pom
    except:
        return default

def save_checkpoint(value):
    checkpoint_path.write_text(str(value),encoding="utf-8")

def get_cookies():
    cookie=os.getenv("STEAM_LOGIN_SECURE", "").strip()
    cookies=dict()
    if cookie:
        cookies["steamLoginSecure"]=cookie
    return cookies

def request_with_backoff(url, params, headers, cookies, retries=max_retries, timeout=30):
    attempt=0
    while attempt<=retries:
        try:
            resp=requests.get(url, params=params, headers=headers, cookies=cookies, timeout=timeout)
            if resp.status_code==429 or resp.status_code==503:
                retry_after=resp.headers.get("Retry-After")
                if retry_after:
                    try:
                        wait=max(1.0, float(retry_after))
                    except Exception:
                        wait=5.0
                else:
                    wait=min(60.0, 2.0**attempt)
                time.sleep(wait)
                attempt+=1
                continue
            return resp

        except requests.RequestException as e:
            wait=min(60.0, 2.0**attempt)
            time.sleep(wait)
            attempt+=1
    return None

def load_items():
    with fai_out.open("r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip()]

def price_to_float(price_str):
    if price_str is None:
        return None
    s=str(price_str).strip()

    if "," in s:
        has_comma=True
    else:
        has_comma=False

    if "." in s:
        has_dot=True
    else:
        has_dot=False

    s="".join(ch for ch in s if ch.isdigit() or ch in ",.-")
    if has_comma and has_dot:
        if s.rfind(".")>s.rfind(","):
            s=s.replace(",", "")
        else:
            s=s.replace(".", "").replace(",", ".")
    elif has_comma and not has_dot:
        s=s.replace(",", ".")
    try:
        return float(s)
    except Exception:
        return None

def parse_row(row):
    if not row or len(row)<2:
        return None, None, None

    raw_ts=row[0]
    dt=None

    if isinstance(raw_ts, (int,float)):
        try:
            dt=datetime.utcfromtimestamp(raw_ts)
        except:
            dt=None
    else:
        s=str(raw_ts).strip()
        if s.endswith("+0"):
            s=s[:-2].strip()
        if s.endswith(":"):
            s=s+"00:00"
        try:
            dt=dateparser.parse(s)
        except:
            dt=None

    if dt is None:
        dt=datetime.utcnow()

    price=price_to_float(row[1])
    vol=None
    if len(row)>2 and row[2] is not None:
        try:
            vol=int(row[2])
        except:
            vol=None

    return dt, price, vol

def insert_batch(conn, rows):
    if not rows:
        return
    with conn.cursor() as cur:
        psycopg2.extras.execute_values(
            cur,
            """
            INSERT INTO market_prices
              (item_name, "timestamp", median_price, volume, currency, appid, retrieved_at)
            VALUES %s
            ON CONFLICT (item_name, "timestamp") DO NOTHING
            """,
            rows,
            page_size=200
        )
    conn.commit()

def fetch_price_history(item_name, appid_local=appid):
    params={"appid": str(appid_local), "market_hash_name": item_name, "currency": 1}
    headers={"User-Agent": agent, "Accept":"application/json"}
    cookies=get_cookies()

    resp=request_with_backoff(steam_url, params=params, headers=headers, cookies=cookies)
    if resp is None:
        return []

    try:
        data=resp.json()
    except:
        return []

    if not data.get("success", True):
        return []

    prices=data.get("prices") or []
    return prices

def get_last_timestamp(cur, item_name):
    cur.execute('SELECT MAX("timestamp") FROM market_prices WHERE item_name=%s', (item_name,))
    row=cur.fetchone()
    return row[0]

def load_start_index():
    return load_checkpoint()

def save_start_index(i):
    save_checkpoint(i)

def main():
    items=load_items()
    if not items:
        print("No items in all_items.txt")
        return

    if not get_cookies():
        print("STEAM_LOGIN_SECURE not set.")
        return
    conn=psycopg2.connect(db_conn)
    total_inserted_this_run=0
    start_index=load_start_index()
    if start_index<0 or start_index>=len(items):
        start_index=0

    if start_index>0:
        print(f"Checkpoint index {start_index} (item {start_index+1}/{len(items)})")

    consecutive_failures=0

    try:
        with conn.cursor() as cur_check:
            n=len(items)
            for idx in range(start_index, n):
                item=items[idx]
                last_ts=get_last_timestamp(cur_check, item)
                hist=fetch_price_history(item)

                if not hist:
                    consecutive_failures+=1
                else:
                    consecutive_failures=0

                if consecutive_failures>=10:
                    save_start_index(idx)
                    break

                batch=list()
                new_points_for_item=0

                for row in hist:
                    dt, price, vol=parse_row(row)
                    if dt is None:
                        continue
                    if last_ts is not None and dt<=last_ts:
                        continue
                    now=datetime.utcnow()
                    batch.append((item, dt, price, vol, curr, appid, now))
                    new_points_for_item+=1
                    if len(batch)>=batch_size:
                        insert_batch(conn, batch)
                        total_inserted_this_run+=len(batch)
                        batch=[]

                if batch:
                    insert_batch(conn, batch)
                    total_inserted_this_run+=len(batch)

                print(f"[{idx+1}/{n}] updated: {item} | "f"rows_inserted_this_run={total_inserted_this_run} | "f"new_points={new_points_for_item} | hist_points={len(hist)}")

                if idx%50==0:
                    save_start_index(idx+1)

                time.sleep(sleep_between_items)
            else:
                save_start_index(0)
    finally:
        conn.close()
    print(f"Total rows inserted: {total_inserted_this_run}")

if __name__=="__main__":
    main()
