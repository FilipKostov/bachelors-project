import math
from pathlib import Path
import os
import numpy as np
import pandas as pd

model_subset=Path("F:/steam_market_project/data/model_subset_csv")
out_subset=Path("F:/steam_market_project/data/model_subset_csv_fe_no_cat")
out_subset.mkdir(parents=True,exist_ok=True)

DATE_COL="date"
ITEM_COL="item_name"
PRICE_COL="steam_price_ffill7"
RAW_PRICE_COL="median_price"
max_files=None

def list_csv_files():
    all_files=os.listdir(model_subset)
    files=[f for f in all_files if f.startswith("subset_") and f.endswith(".csv")]
    files.sort()
    if max_files is not None:
        files=files[:max_files]
    return files

def safe_div(a,b):
    out=list()
    for i in range(len(a)):
        try:
            if b[i]==0 or b[i] is None:
                out.append(None)
            else:
                out.append(a[i]/b[i])
        except:
            out.append(None)
    return out

def per_item_days_since_trade(df):
    valid=df["steam_valid"]
    out=list()
    last_seen=-1
    for i in range(len(valid)):
        if valid.iloc[i]==1:
            out.append(0)
            last_seen=i
        else:
            out.append(i-last_seen if last_seen>=0 else 999999)
    return out

def add_gap_features(df):
    df=df.copy()
    if "volume" in df.columns:
        df["steam_valid"]=(pd.to_numeric(df["volume"],errors="coerce").fillna(0)>0).astype(int)
    else:
        if RAW_PRICE_COL in df.columns:
            df["steam_valid"]=df[RAW_PRICE_COL].notna().astype(int)
        else:
            df["steam_valid"]=1

    if RAW_PRICE_COL in df.columns:
        df["missing_raw"]=df[RAW_PRICE_COL].isna().astype(int)
    else:
        df["missing_raw"]=0
    df["price_was_filled"]=((df["steam_valid"]==0) & (df[PRICE_COL].notna())).astype(int)
    df=df.sort_values([ITEM_COL,DATE_COL],kind="mergesort").reset_index(drop=True)
    chunks=list()
    for item,sub in df.groupby(ITEM_COL,sort=False):
        sub=sub.copy()
        sub["days_since_last_trade"]=per_item_days_since_trade(sub)
        chunks.append(sub)
    df_out=pd.concat(chunks,ignore_index=True)
    df_out=df_out.sort_values([DATE_COL,ITEM_COL],kind="mergesort").reset_index(drop=True)
    return df_out

def add_more_rolling(df):
    df=df.copy()
    df=df.sort_values([ITEM_COL,DATE_COL],kind="mergesort").reset_index(drop=True)
    df[PRICE_COL]=pd.to_numeric(df[PRICE_COL],errors="coerce")

    chunks=list()
    for item,sub in df.groupby(ITEM_COL,sort=False):
        sub=sub.copy()
        p=sub[PRICE_COL].astype(float)

        sub["mom_7"]=pd.Series(safe_div(p,p.shift(7)))-1.0
        sub["mom_14"]=pd.Series(safe_div(p,p.shift(14)))-1.0
        sub["roll_mean_14"]=p.rolling(14,min_periods=5).mean()
        sub["roll_std_14"]=p.rolling(14,min_periods=5).std()
        sub["log_price"]=np.log1p(np.clip(p,0,None))

        sub=sub.iloc[14:]
        chunks.append(sub)

    df_out=pd.concat(chunks,ignore_index=True)
    df_out=df_out.sort_values([DATE_COL,ITEM_COL],kind="mergesort").reset_index(drop=True)
    return df_out

def main():
    files=list_csv_files()
    for i,f in enumerate(files,1):
        df=pd.read_csv(model_subset/f)
        df[DATE_COL]=pd.to_datetime(df[DATE_COL],errors="coerce")
        df=df.dropna(subset=[DATE_COL])

        if PRICE_COL not in df.columns:
            raise RuntimeError(f"missing")
        df=add_gap_features(df)
        df=add_more_rolling(df)
        out_path=out_subset/Path(f).name
        df.to_csv(out_path,index=False)

if __name__=="__main__":
    main()