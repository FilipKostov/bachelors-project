import time
import random
from pathlib import Path
from calendar import monthrange
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor

db_conn="dbname=steam user=postgres password=password host=127.0.0.1 port=5432"
table="market_prices"

out_dir="F:/steam_market_project/data/model_subset_csv"
checkpoint="F:/steam_market_project/data/model_subset_csv_checkpoint.txt"

TS_COL='"timestamp"'
RETRIEVED_COL='"retrieved_at"'
PRICE_COL_RAW="median_price"
VOL_COL_RAW="volume"

ffill_limit_days=60

lags=[1,7,30]
roll_windows=[7,30]
target_days=7

lookback=30
lookahead=target_days

ITEM_CHUNK=25
sleep_after=0.05

manual_items=["★ Talon Knife | Crimson Web","AK-47 | Asiimov","Desert Eagle | Code Red","AWP | Asiimov","Fracture Case","Dreams & Nightmares Case","Operation Broken Fang Case","Revolution Case"]

ITEM_DIM_CSV="F:/steam_market_project/data/item_dim.csv"

DESIRED_TOTAL_ITEMS=950

ITEM_DIM=None

def load_checkpoint():
    try:
        return Path(checkpoint).read_text().strip()
    except:
        return ""

def save_checkpoint(s):
    Path(checkpoint).write_text(str(s))

def month_start_end(y,m):
    start=pd.Timestamp(y,m,1)
    end=pd.Timestamp(y,m,monthrange(y,m)[1])
    return start,end

def list_months(start_date,end_date):
    s=pd.to_datetime(start_date)
    e=pd.to_datetime(end_date)
    out=list()
    cur=pd.Timestamp(s.year,s.month,1)
    while cur<=e:
        out.append((cur.year,cur.month))
        if cur.month==12:
            cur=pd.Timestamp(cur.year+1,1,1)
        else:
            cur=pd.Timestamp(cur.year,cur.month+1,1)
    return out

def load_dim_or_none():
    p=Path(ITEM_DIM_CSV)
    if not p.exists():
        return None
    dim=pd.read_csv(p)
    if "item_name" not in dim.columns:
        return None
    dim["item_name"]=dim["item_name"].astype(str)
    return dim

def get_db_max_retrieved(conn):
    query=f"SELECT MAX({RETRIEVED_COL}) AS rt FROM {table}"
    cur=conn.cursor()
    cur.execute(query)
    row=cur.fetchone()
    cur.close()
    return pd.to_datetime(row[0])

def choose_subset_items(conn):
    query=f"SELECT DISTINCT item_name FROM {table} ORDER BY item_name"
    df=pd.read_sql(query,conn)
    all_items=df["item_name"].tolist()
    manual_present=[x for x in manual_items if x in all_items]
    remaining=[x for x in all_items if x not in manual_present]
    target_random=max(0,DESIRED_TOTAL_ITEMS-len(manual_present))
    if len(remaining)<=target_random:
        random_selected=remaining
    else:
        random_selected=random.sample(remaining,target_random)
    subset=sorted(set(manual_present+random_selected))
    return subset

def fetch_chunk(conn,items,start_date,end_date):
    query=f"""
    SELECT 
        item_name, 
        {TS_COL} AS ts, 
        {PRICE_COL_RAW} AS price, 
        COALESCE({VOL_COL_RAW},0) AS vol 
    FROM {table} 
    WHERE ({TS_COL}::date)>= %s 
      AND ({TS_COL}::date)<= %s 
      AND item_name=ANY(%s) 
    ORDER BY item_name,ts
    """
    cur=conn.cursor(cursor_factory=RealDictCursor)
    cur.execute(query,(start_date,end_date,items))
    rows=cur.fetchall()
    cur.close()
    return pd.DataFrame(rows)

def make_daily_panel(df):
    if len(df)==0:
        return df
    df["ts"]=pd.to_datetime(df["ts"],errors="coerce")
    df=df[df["ts"].notna()].copy()
    if len(df)==0:
        return df
    df["date"]=df["ts"].dt.floor("D")
    df["price"]=pd.to_numeric(df["price"],errors="coerce")
    df["vol"]=pd.to_numeric(df["vol"],errors="coerce").fillna(0.0)
    grouped=df.groupby(["item_name","date"],as_index=False).agg(median_price=("price","median"),volume=("vol","sum"))
    grouped=grouped.sort_values(["item_name","date"],kind="mergesort").reset_index(drop=True)
    return grouped

def add_ffill_price(daily,start_date,end_date):
    if len(daily)==0:
        return daily
    start_date=pd.to_datetime(start_date)
    end_date=pd.to_datetime(end_date)
    daily=daily.sort_values(["item_name","date"],kind="mergesort").reset_index(drop=True)
    out=list()
    for name,grp in daily.groupby("item_name"):
        d0=max(start_date,grp["date"].min())
        d1=min(end_date,grp["date"].max())
        if d0>d1:
            continue
        cal=pd.date_range(d0,d1,freq="D")
        tmp=pd.DataFrame({"date":cal})
        price_dict=dict(zip(grp["date"],grp["median_price"]))
        vol_dict=dict(zip(grp["date"],grp["volume"]))
        tmp["item_name"]=name
        tmp["median_price"]=[price_dict.get(d,float("nan")) for d in cal]
        tmp["steam_price_ffill7"]=tmp["median_price"].copy()
        last_val=float("nan")
        for i in range(len(tmp)):
            if pd.notna(tmp.loc[i,"steam_price_ffill7"]):
                last_val=tmp.loc[i,"steam_price_ffill7"]
            elif pd.isna(tmp.loc[i,"steam_price_ffill7"]) and pd.notna(last_val):
                tmp.loc[i,"steam_price_ffill7"]=last_val
        tmp["volume"]=[vol_dict.get(d,0.0) for d in cal]
        out.append(tmp)
    if not out:
        return pd.DataFrame(columns=["date","item_name","median_price","volume","steam_price_ffill7"])
    out=pd.concat(out,ignore_index=True)
    out=out.sort_values(["item_name","date"],kind="mergesort").reset_index(drop=True)
    return out

def add_features_and_target(df):
    if len(df)==0:
        return df
    df=df.sort_values(["item_name","date"],kind="mergesort").reset_index(drop=True)
    item_names=list(set(df["item_name"]))
    for k in lags:
        df["p_lag"+str(k)]=float("nan")
        for name in item_names:
            grp_idx=df[df["item_name"]==name].index.tolist()
            for i in range(len(grp_idx)):
                if i>=k:
                    df.loc[grp_idx[i],"p_lag"+str(k)]=df.loc[grp_idx[i-k],"steam_price_ffill7"]
    df["ret_1"]=float("nan")
    df["ret_7"]=float("nan")
    for name in item_names:
        grp_idx=df[df["item_name"]==name].index.tolist()
        for i in grp_idx:
            if pd.notna(df.loc[i,"p_lag1"]):
                df.loc[i,"ret_1"]=df.loc[i,"steam_price_ffill7"]/df.loc[i,"p_lag1"]-1.0
            if pd.notna(df.loc[i,"p_lag7"]):
                df.loc[i,"ret_7"]=df.loc[i,"steam_price_ffill7"]/df.loc[i,"p_lag7"]-1.0
    for w in roll_windows:
        if w==7:
            minp=3
        else:
            minp=10
        col_mean="roll_mean_"+str(w)
        col_std="roll_std_"+str(w)
        df[col_mean]=float("nan")
        df[col_std]=float("nan")
        for name in item_names:
            grp_idx=df[df["item_name"]==name].index.tolist()
            prices=[df.loc[i,"steam_price_ffill7"] for i in grp_idx]
            for i in range(len(grp_idx)):
                window=prices[max(0,i-w+1):i+1]
                vals=[x for x in window if pd.notna(x)]
                if len(vals)>=minp:
                    mean_val=sum(vals)/len(vals)
                    df.loc[grp_idx[i],col_mean]=mean_val
                    var=sum((x-mean_val)**2 for x in vals)/len(vals)
                    df.loc[grp_idx[i],col_std]=var**0.5
    df["roll_vol_30"]=float("nan")
    for name in item_names:
        grp_idx=df[df["item_name"]==name].index.tolist()
        ret1=[df.loc[i,"ret_1"] for i in grp_idx]
        for i in range(len(grp_idx)):
            window=ret1[max(0,i-29):i+1]
            vals=[x for x in window if pd.notna(x)]
            if len(vals)>=10:
                mean_val=sum(vals)/len(vals)
                var=sum((x-mean_val)**2 for x in vals)/len(vals)
                df.loc[grp_idx[i],"roll_vol_30"]=var**0.5
    df["vol_sum_7"]=float("nan")
    df["vol_mean_7"]=float("nan")
    df["vol_sum_30"]=float("nan")
    df["vol_mean_30"]=float("nan")
    for name in item_names:
        grp_idx=df[df["item_name"]==name].index.tolist()
        vols=[df.loc[i,"volume"] for i in grp_idx]
        for i in range(len(grp_idx)):
            w7=vols[max(0,i-6):i+1]
            v7=[x for x in w7 if pd.notna(x)]
            if len(v7)>=3:
                df.loc[grp_idx[i],"vol_sum_7"]=sum(v7)
                df.loc[grp_idx[i],"vol_mean_7"]=sum(v7)/len(v7)
            w30=vols[max(0,i-29):i+1]
            v30=[x for x in w30 if pd.notna(x)]
            if len(v30)>=10:
                df.loc[grp_idx[i],"vol_sum_30"]=sum(v30)
                df.loc[grp_idx[i],"vol_mean_30"]=sum(v30)/len(v30)
    df["dow"]=[pd.to_datetime(df.loc[i,"date"]).dayofweek for i in range(len(df))]
    df["month_num"]=[pd.to_datetime(df.loc[i,"date"]).month for i in range(len(df))]
    df["gap_days"]=float("nan")
    for name in item_names:
        grp_idx=df[df["item_name"]==name].index.tolist()
        streak=0
        for i in grp_idx:
            if pd.isna(df.loc[i,"steam_price_ffill7"]):
                streak=0
            else:
                if pd.isna(df.loc[i,"median_price"]):
                    streak+=1
                else:
                    streak=0
                df.loc[i,"gap_days"]=min(streak,60)
    df_desc=df.sort_values(["item_name","date"],ascending=[True,False],kind="mergesort").reset_index(drop=True)
    df_desc["y_avg_7d_fwd"]=float("nan")
    for name in item_names:
        grp_idx=df_desc[df_desc["item_name"]==name].index.tolist()
        prices=[df_desc.loc[i,"steam_price_ffill7"] for i in grp_idx]
        for i in range(len(grp_idx)):
            window=prices[i+1:i+8]
            vals=[x for x in window if pd.notna(x)]
            if len(vals)>=1:
                df_desc.loc[grp_idx[i],"y_avg_7d_fwd"]=sum(vals)/len(vals)
    df["y_avg_7d_fwd"]=df_desc.sort_index()["y_avg_7d_fwd"].values
    if ITEM_DIM is not None:
        df=df.merge(ITEM_DIM,how="left",on="item_name")
        cols_to_drop=["item_type","weapon_class","rarity","collection","quality","wear_full","sticker_type"]
        existing_to_drop=[c for c in cols_to_drop if c in df.columns]
        if existing_to_drop:
            df=df.drop(columns=existing_to_drop)
    df["item_name"]=df["item_name"].astype("string")
    df["steam_price_ffill7"]=pd.to_numeric(df["steam_price_ffill7"],errors="coerce").astype("float64")
    df["volume"]=pd.to_numeric(df["volume"],errors="coerce").astype("float64")
    df["y_avg_7d_fwd"]=pd.to_numeric(df["y_avg_7d_fwd"],errors="coerce").astype("float64")
    df["gap_days"]=pd.to_numeric(df["gap_days"],errors="coerce").astype("float64")
    return df

def main():
    global ITEM_DIM
    Path(out_dir).mkdir(parents=True,exist_ok=True)
    ITEM_DIM=load_dim_or_none()
    last_done=load_checkpoint()
    conn=psycopg2.connect(db_conn)
    subset_items=choose_subset_items(conn)
    db_min_retrieved=get_db_max_retrieved(conn)
    common_end_date=db_min_retrieved.floor("D").date().isoformat()
    query_min=f"SELECT MIN({TS_COL}::date) AS min_date FROM {table}"
    min_date_df=pd.read_sql(query_min,conn)
    global_start=min_date_df["min_date"].iloc[0]
    months=list_months(global_start,common_end_date)
    start_idx=0
    if last_done:
        for i in range(len(months)):
            y,m=months[i]
            if f"{y}-{m:02d}"==last_done:
                start_idx=i+1
                break
    for mi in range(start_idx,len(months)):
        y,m=months[mi]
        key=f"{y}-{m:02d}"
        out_file=Path(out_dir)/f"subset_{y}_{m:02d}.csv"
        tmp_file=Path(out_dir)/f"subset_{y}_{m:02d}.tmp.csv"
        try:
            m_start,m_end=month_start_end(y,m)
            m_start=min(m_start,pd.to_datetime(common_end_date))
            m_end=min(m_end,pd.to_datetime(common_end_date))
            if m_start>pd.to_datetime(common_end_date):
                save_checkpoint(key)
                continue
            q_start=(m_start-pd.Timedelta(days=lookback)).date().isoformat()
            q_end=(m_end+pd.Timedelta(days=lookahead)).date().isoformat()
            if q_end>common_end_date:
                q_end=common_end_date
            chunks=[subset_items[i:i+ITEM_CHUNK] for i in range(0,len(subset_items),ITEM_CHUNK)]
            first=True
            for chunk in chunks:
                raw=fetch_chunk(conn,chunk,q_start,q_end)
                if len(raw)==0:
                    del raw
                    time.sleep(sleep_after)
                    continue
                daily=make_daily_panel(raw)
                del raw
                if len(daily)==0:
                    del daily
                    time.sleep(sleep_after)
                    continue
                panel=add_ffill_price(daily,q_start,q_end)
                del daily
                if len(panel)==0:
                    del panel
                    time.sleep(sleep_after)
                    continue
                feat=add_features_and_target(panel)
                del panel
                if len(feat)==0:
                    del feat
                    time.sleep(sleep_after)
                    continue
                feat=feat[(feat["date"]>=m_start)&(feat["date"]<=m_end)]
                feat=feat[feat["steam_price_ffill7"].notna()]
                if len(feat)==0:
                    del feat
                    time.sleep(sleep_after)
                    continue
                mode="w" if first else "a"
                feat.to_csv(tmp_file,index=False,mode=mode,header=first)
                first=False
                del feat
                time.sleep(sleep_after)
        finally:
            if tmp_file.exists():
                tmp_file.replace(out_file)
            save_checkpoint(key)
    save_checkpoint("")
    conn.close()
if __name__=="__main__":
    main()