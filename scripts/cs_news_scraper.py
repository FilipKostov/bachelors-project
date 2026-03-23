import os
import time
import pandas as pd
from datetime import datetime,timezone
import requests
import psycopg2
import psycopg2.extras
from pathlib import Path

data_dir="F:/steam_market_project/data"
fe_dir="F:/steam_market_project/data/model_subset_csv_fe"
ext_dir="F:/steam_market_project/data/external"
fe_ext_dir="F:/steam_market_project/data/model_subset_csv_fe_ext"

app_id=730
url="https://api.steampowered.com/ISteamNews/GetNewsForApp/v0002/"
feeds="steam_community_announcements"
count=200
max_calls=5000
sleep_sec=0.35
timeout=30
ua="Mozilla/5.0 (Windows NT 10.0; Win64; x64)"

db_conn_str="dbname=steam user=postgres password=password host=127.0.0.1 port=5432"

kw_rules={"kw_update":["update","patch","release notes","releasenotes","bug fix","fixes"],"kw_trade":["trade","trading","trade hold","hold","reversal","steam guard","authenticator","phone","verification"],"kw_market":["market","community market","steam market"],"kw_case":["case","capsule","collection","sticker"],"kw_rank":["rank","rating","premier","leaderboard","elo","season"],"kw_vac":["vac","anti-cheat","anticheat","ban"],"kw_map":["map","inferno","mirage","nuke","overpass","ancient","vertigo","anubis"]}

pom={"t0":(0,0),"t7":(0,7),"pm7":(-7,7),"pm30":(-30,30)}

CHECKPOINT_FILE=Path(ext_dir)/"external_news_checkpoint.txt"
OUT_PATTERN="external_daily_{year}_{month:02d}.csv"
IDX_PATTERN="idx_news_{year}_{month:02d}.csv"

def load_checkpoint():
    if CHECKPOINT_FILE.exists():
        return CHECKPOINT_FILE.read_text().strip()
    return ""

def save_checkpoint(val):
    CHECKPOINT_FILE.write_text(str(val))

def get_all_fe_dates():
    dates=set()
    for f in Path(fe_dir).glob("subset_*.csv"):
        try:
            df=pd.read_csv(f,usecols=["date"],dtype={"date":str})
            dates.update(df["date"].dropna().unique())
        except:
            pass
    if not dates:
        return None
    return max(dates)

def unix_to_date(ts):
    return datetime.fromtimestamp(int(ts),tz=timezone.utc).date().isoformat()

def fetch_news_page(enddate_unix=None):
    params={"appid":app_id,"count":count,"maxlength":0,"format":"json","feeds":feeds}
    if enddate_unix:
        params["enddate"]=int(enddate_unix)
    r=requests.get(url,params=params,headers={"User-Agent":ua},timeout=timeout)
    r.raise_for_status()
    return r.json().get("appnews",{}).get("newsitems",[])

def normalize_news(items):
    out=list()
    for it in items:
        ts=it.get("date")
        if ts is None:
            continue
        out.append({"gid":str(it.get("gid","")),"title":str(it.get("title","")),"author":str(it.get("author","")),"feedlabel":str(it.get("feedlabel","")),"date_unix":int(ts),"date":unix_to_date(ts),"url":str(it.get("url","")),"is_external_url":int(bool(it.get("is_external_url",False))),"contents":str(it.get("contents",""))})
    return out

def backfill_all_news(max_date_str):
    max_dt=pd.to_datetime(max_date_str)
    all_rows=list()
    seen=set()
    enddate_unix=None
    calls=0
    while calls<max_calls:
        calls+=1
        items=fetch_news_page(enddate_unix)
        if not items:
            break
        rows=normalize_news(items)
        min_ts=None
        added=0
        for r in rows:
            key=r["gid"]or(r["url"],r["date_unix"],r["title"])
            if key in seen:
                continue
            seen.add(key)
            d=pd.to_datetime(r["date"])
            if d>max_dt:
                continue
            all_rows.append(r)
            added+=1
            if min_ts is None or r["date_unix"]<min_ts:
                min_ts=r["date_unix"]
        print(f"call {calls:3d} | got {len(items):3d} | new {added:4d} | enddate {enddate_unix}")
        if min_ts is None or added==0:
            break
        enddate_unix=min_ts-1
        time.sleep(sleep_sec)
    if not all_rows:
        return pd.DataFrame()
    df=pd.DataFrame(all_rows)
    df=df.sort_values(["date_unix","gid"]).reset_index(drop=True)
    df["date"]=pd.to_datetime(df["date"]).dt.date.astype(str)
    return df

def has_any(text,words):
    t=str(text).lower()
    return any(w in t for w in words)

def build_daily_flags(posts,date_range):
    if posts.empty:
        base=pd.DataFrame({"date":date_range.date.astype(str)})
        base["cs2_post_cnt"]=0
        for k in kw_rules:
            base[k]=0
        return base
    txt=(posts["title"].fillna("")+"\n"+posts["contents"].fillna("")).astype(str)
    tmp=posts[["date"]].copy()
    tmp["cs2_post_cnt"]=1
    for k,words in kw_rules.items():
        tmp[k]=[1 if has_any(t,words) else 0 for t in txt]
    daily=tmp.groupby("date",as_index=False).sum(numeric_only=True)
    base=pd.DataFrame({"date":date_range.date.astype(str)})
    out=base.merge(daily,on="date",how="left").fillna(0)
    for c in out.columns:
        if c!="date":
            out[c]=out[c].astype("int32")
    return out

def load_market_daily_avg():
    conn=psycopg2.connect(db_conn_str)
    df=pd.read_sql("""
        SELECT timestamp::date AS date, AVG(median_price) AS avg_price
        FROM market_prices
        GROUP BY timestamp::date
        ORDER BY date
    """,conn)
    conn.close()
    df["date"]=pd.to_datetime(df["date"]).dt.date.astype(str)
    return df.set_index("date")["avg_price"].to_dict()

def compute_update_impact(posts,daily_avg):
    posts=posts.copy()
    posts["update_impact_score"]=0.0
    for i,row in posts.iterrows():
        if not has_any(row["title"],kw_rules["kw_update"]):
            continue
        d=pd.to_datetime(row["date"]).date()
        before=[daily_avg.get((d-pd.Timedelta(days=x)).isoformat(),0) for x in range(7,0,-1)]
        after=[daily_avg.get((d+pd.Timedelta(days=x)).isoformat(),0) for x in range(1,8)]
        if not before or not after or sum(before)==0:
            continue
        avg_b=sum(before)/len(before)
        avg_a=sum(after)/len(after)
        rel_change=abs(avg_a-avg_b)/avg_b
        score=min(rel_change*5,5)/5
        posts.at[i,"update_impact_score"]=score
    return posts

def build_idx_windows(daily,posts=None,market_avg=None):
    out=pd.DataFrame({"date":pd.to_datetime(daily["date"])})
    daily=daily.copy()
    daily["date"]=pd.to_datetime(daily["date"], errors="coerce")
    daily = daily.dropna(subset=["date"])
    for kw in kw_rules:
        event_days=daily[daily[kw]==1]["date"].dt.date.tolist()
        impact_dict=None
        if kw=="kw_update" and posts is not None and market_avg:
            scored=compute_update_impact(posts,market_avg)
            impact_dict=dict(zip(scored["date"],scored["update_impact_score"]))
        for wname,(a,b) in pom.items():
            col=f"idx_{kw.replace('kw_','')}_{wname}"
            out[col]=0.0
            for d in event_days:
                lo=pd.Timestamp(d)+pd.Timedelta(days=a)
                hi=pd.Timestamp(d)+pd.Timedelta(days=b)
                mask=(out["date"]>=lo)&(out["date"]<=hi)
                score=impact_dict.get(d.isoformat(),1.0) if impact_dict else 1.0
                out.loc[mask,col]=score
    out["date"]=out["date"].dt.date.astype(str)
    return out

def process_month(year,month,posts_all,market_avg):
    m_start=pd.Timestamp(year,month,1)
    m_end=m_start+pd.offsets.MonthEnd(0)
    cal=pd.date_range(m_start,m_end,freq="D")

    month_posts=posts_all[(pd.to_datetime(posts_all["date"])>=m_start)&(pd.to_datetime(posts_all["date"])<=m_end)]
    daily=build_daily_flags(month_posts,cal)
    daily["date"]=pd.to_datetime(daily["date"])
    idx=build_idx_windows(daily,month_posts,market_avg)
    idx["date"]=pd.to_datetime(idx["date"])
    merged=daily.merge(idx,on="date",how="left")
    merged["date"]=merged["date"].dt.date.astype(str)

    for c in merged.columns:
        if c=="date":
            continue
        merged[c]=pd.to_numeric(merged[c],errors="coerce").fillna(0).astype("float32")

    return merged,idx

def main():
    max_date_str=get_all_fe_dates()
    if not max_date_str:
        return
    last_done=load_checkpoint()
    posts_all=backfill_all_news(max_date_str)
    posts_all.to_csv(Path(ext_dir)/"cs2_steamnews_posts_all.csv",index=False)
    market_avg=load_market_daily_avg()
    months=list()
    cur=pd.Timestamp(2013,1,1)
    end=pd.to_datetime(max_date_str)+pd.Timedelta(days=32)
    while cur<=end:
        months.append((cur.year,cur.month))
        cur+=pd.offsets.MonthBegin(1)
    start_i=0
    if last_done:
        for i,(y,m) in enumerate(months):
            if f"{y}-{m:02d}"==last_done:
                start_i=i+1
                break
    for i in range(start_i,len(months)):
        y,m=months[i]
        key=f"{y}-{m:02d}"
        daily,idx=process_month(y,m,posts_all,market_avg)
        daily.to_csv(Path(ext_dir)/OUT_PATTERN.format(year=y,month=m),index=False)
        idx.to_csv(Path(ext_dir)/IDX_PATTERN.format(year=y,month=m),index=False)
        save_checkpoint(key)
        print(f"  saved {len(daily)} rows")
    save_checkpoint("")
if __name__=="__main__":
    main()