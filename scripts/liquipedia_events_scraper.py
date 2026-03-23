import os
import pandas as pd
import requests
from datetime import datetime
from pathlib import Path
import re

data_dir="F:/steam_market_project/data"
fe_dir="F:/steam_market_project/data/model_subset_csv_fe"
ext_dir="F:/steam_market_project/data/external"
out_dir="F:/steam_market_project/data/model_subset_csv_raw_liquipedia"
base_url="https://liquipedia.net/counterstrike"
date_col="date"
LIQUIPEDIA_URLS={"post_2023":"https://liquipedia.net/counterstrike/S-Tier_Tournaments/Post_2023","2012_2023":"https://liquipedia.net/counterstrike/S-Tier_Tournaments/2012-2023"}
pom={"t0":(0,0),"t7":(0,7),"pm7":(-7,7),"pm30":(-30,30)}
months_map={"jan":1,"january":1,"feb":2,"february":2,"mar":3,"march":3,"apr":4,"april":4,"may":5,"jun":6,"june":6,"jul":7,"july":7,"aug":8,"august":8,"sep":9,"sept":9,"september":9,"oct":10,"october":10,"nov":11,"november":11,"dec":12,"december":12}

CHECKPOINT_FILE=Path(ext_dir)/"liquipedia_checkpoint.txt"
OUT_PATTERN="liquipedia_daily_{year}_{month:02d}.csv"
IDX_PATTERN="idx_liquipedia_{year}_{month:02d}.csv"

def month_to_int(s):
    return months_map.get(str(s).strip().lower())

def get_all_fe_dates():
    dates=set()
    for f in Path(fe_dir).glob("subset_*.csv"):
        try:
            df=pd.read_csv(f,usecols=[date_col],dtype={date_col:str})
            dates.update(df[date_col].dropna().unique())
        except:
            pass
    return max(dates) if dates else None

def extract_ranges_from_text(txt):
    txt=txt.replace('&ndash;','-').replace('&mdash;','-').replace('\u2013','-').replace('\u2014','-').replace('–','-')
    txt=re.sub(r'\s+',' ',txt.strip())
    out=list()
    date_pattern=r'([A-Za-z]{3,})\s*(\d{1,2})\s*[-–—]\s*(\d{1,2})\s*,\s*(\d{4})'
    matches=re.findall(date_pattern,txt,re.IGNORECASE)
    for mo_str,d1_str,d2_str,year_str in matches:
        try:
            mo_str_lower=mo_str.lower()[:3]
            mo=month_to_int(mo_str_lower)
            if not mo:
                continue
            d1=int(d1_str)
            d2=int(d2_str)
            year=int(year_str)
            st=datetime(year,mo,d1).date()
            en=datetime(year,mo,d2).date()
            if en<st:
                st,en=en,st
            out.append((st,en))
        except ValueError:
            pass
    return list(set(out))

def load_ranges_from_liquipedia():
    all_ranges=list()
    for url in LIQUIPEDIA_URLS.values():
        try:
            res=requests.get(url,headers={"User-Agent":"Mozilla/5.0"})
            res.raise_for_status()
            ranges=extract_ranges_from_text(res.text)
            all_ranges.extend(ranges)
        except:
            pass
    return list(set(all_ranges))

def mark_ranges(base,ranges,col,weight=1):
    out=base.copy()
    out[col]=0
    out[col+"_cnt"]=0.0
    dts=pd.to_datetime(base["date"])
    for st,en in ranges:
        mask=(dts>=pd.Timestamp(st))&(dts<=pd.Timestamp(en))
        out.loc[mask,col+"_cnt"]+=weight
    out[col+"_cnt"]=out[col+"_cnt"].clip(upper=10.0)
    out[col]=(out[col+"_cnt"]>0).astype("int32")
    return out

def build_idx_windows(daily,cols):
    out=daily[["date"]].copy()
    for c in cols:
        base=daily[c].astype(int).values
        for name,(a,b) in pom.items():
            arr=[0]*len(base)
            hits=[i for i,v in enumerate(base) if v==1]
            for i in hits:
                for j in range(max(0,i+a),min(len(base),i+b+1)):
                    arr[j]=1
            out["idx_"+c+"_"+name]=pd.Series(arr,dtype="int32")
    return out

def build_daily_for_month(year,month,all_ranges):
    m_start=pd.Timestamp(year,month,1)
    m_end=m_start+pd.offsets.MonthEnd(0)
    base=pd.DataFrame({"date":pd.date_range(m_start,m_end,freq="D").date.astype(str)})
    daily=base.copy()
    daily=mark_ranges(daily,all_ranges,"lp_valve_tier1",weight=5)
    daily=mark_ranges(daily,all_ranges,"lp_s_tier",weight=4)
    daily=mark_ranges(daily,all_ranges,"lp_a_tier",weight=2)
    daily["lp_event_score"]=(5*daily["lp_valve_tier1_cnt"]+4*daily["lp_s_tier_cnt"]+2*daily["lp_a_tier_cnt"]).astype("int32")
    daily["lp_event_intensity"]=daily["lp_event_score"]
    for prefix in ["lp_valve_tier1_cnt","lp_s_tier_cnt","lp_a_tier_cnt","lp_event_score","lp_event_intensity"]:
        m=daily[prefix].max()
        daily[prefix+"_norm"]=(daily[prefix]/m).astype("float32") if m>0 else 0.0
    return daily

def load_checkpoint():
    return CHECKPOINT_FILE.read_text().strip() if CHECKPOINT_FILE.exists() else ""

def save_checkpoint(val):
    CHECKPOINT_FILE.write_text(str(val))

def main():
    max_date_str=get_all_fe_dates()
    if not max_date_str:
        return
    last_done=load_checkpoint()
    all_ranges=load_ranges_from_liquipedia()
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
        daily=build_daily_for_month(y,m,all_ranges)
        idx=build_idx_windows(daily,["lp_valve_tier1","lp_s_tier","lp_a_tier"])
        daily.to_csv(Path(ext_dir)/OUT_PATTERN.format(year=y,month=m),index=False)
        idx.to_csv(Path(ext_dir)/IDX_PATTERN.format(year=y,month=m),index=False)
        save_checkpoint(key)
    save_checkpoint("")

if __name__=="__main__":
    main()