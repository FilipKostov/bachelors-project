import psycopg2
import pandas as pd
import matplotlib.pyplot as plt

db_conn="dbname=steam user=postgres password=password host=127.0.0.1 port=5432"

ffill_limit=7
roll_c=30
roll_v=30

items=list((("knife", "★ Talon Knife | Crimson Web"),("ak_red", "AK-47 | Asiimov"),("deagle_red", "Desert Eagle | Printstream"),("awp", "AWP | Asiimov"),("case_common", "Fracture Case"),("case_mid", "Dreams & Nightmares Case"),("case_rare", "Operation Broken Fang Case"),("case_new", "Revolution Case"),("butterfly_ch_fn", "★ Butterfly Knife | Case Hardened")))

def get_global_date_range(conn):
    cur=conn.cursor()
    cur.execute("SELECT MIN(retrieved_at)::date AS min_date, MAX(timestamp)::date AS max_date FROM market_prices")
    row=cur.fetchone()
    cur.close()
    return row[0],row[1]

def pick_best_match(conn,name):
    cur=conn.cursor()
    cur.execute(
        """
        select item_name, coalesce(sum(volume),0) as v, count(*) as n
        from market_prices
        where item_name ilike %s
        group by item_name
        order by v desc, n desc
        limit 1
        """,
        ("%"+name+"%",),
    )
    row=cur.fetchone()
    cur.close()
    if not row:
        return None
    return row[0]

def fetch_all_available(conn,item_name):
    df=pd.read_sql(
        """
        select "timestamp", median_price, volume, retrieved_at
        from market_prices
        where item_name=%s
        order by "timestamp"
        """,
        conn,
        params=[item_name],
    )
    if df.empty:
        return df

    df["timestamp"]=pd.to_datetime(df["timestamp"])
    df["retrieved_at"]=pd.to_datetime(df["retrieved_at"])
    df["date"]=df["timestamp"].dt.floor("D")

    first_retrieved=df["retrieved_at"].min()
    df=df[df["retrieved_at"]>=first_retrieved].copy()

    daily=df.groupby("date",as_index=False).agg(
        steam_price=("median_price","median"),
        steam_volume=("volume","sum"),
        steam_n_points=("median_price","size"),
    )
    return daily

def build_daily_panel(daily,ffill_limit_days=7):
    if daily.empty:
        return pd.DataFrame(),{"total_days":0,"raw_missing":0,"raw_missing_pct":0.0,"ffill_missing":0,"ffill_missing_pct":0.0}

    min_date=daily["date"].min().floor("D")
    max_date=daily["date"].max().ceil("D")

    dates=pd.date_range(min_date,max_date,freq="D")
    out=pd.DataFrame(dict(date=dates))

    out=out.merge(daily,how="left",on="date")
    out["steam_valid"]=out["steam_price"].notna().astype(int)
    out["steam_volume"]=out["steam_volume"].fillna(0)

    out["days_since_last_trade"]=0
    last_seen=None
    for i in range(len(out)):
        if out.loc[i,"steam_valid"]==1:
            last_seen=out.loc[i,"date"]
            out.loc[i,"days_since_last_trade"]=0
        else:
            if last_seen is None:
                out.loc[i,"days_since_last_trade"]=999999
            else:
                out.loc[i,"days_since_last_trade"]=int((out.loc[i,"date"]-last_seen).days)

    out["steam_price_ffill"]=out["steam_price"].ffill()
    out.loc[out["days_since_last_trade"]>ffill_limit_days,"steam_price_ffill"]=pd.NA

    out["price_was_filled"]=((out["steam_valid"]==0)&(out["steam_price_ffill"].notna())).astype(int)
    out["missing_raw"]=out["steam_price"].isna().astype(int)
    out["missing_ffill"]=out["steam_price_ffill"].isna().astype(int)

    total_days=len(out)
    raw_missing_count=out["steam_price"].isna().sum()
    raw_missing_pct=(raw_missing_count/total_days*100)if total_days>0 else 0.0
    ffill_missing_count=out["steam_price_ffill"].isna().sum()
    ffill_missing_pct=(ffill_missing_count/total_days*100)if total_days>0 else 0.0

    summary={"total_days":total_days,"raw_missing":raw_missing_count,"raw_missing_pct":raw_missing_pct,"ffill_missing":ffill_missing_count,"ffill_missing_pct":ffill_missing_pct}

    return out,summary

def sanity(panel,key,ffill_limit_days):
    x=panel.copy()

    missing_gaps=(x["date"].diff().dt.days.dropna()>1).sum()
    vol_bad=(x.loc[x["steam_valid"]==0,"steam_volume"]!=0).sum()
    fill_over_limit=((x["steam_valid"]==0)&(x["days_since_last_trade"]>ffill_limit_days)&(x["steam_price_ffill"].notna())).sum()

    trade_day_missing_price=((x["steam_valid"]==1)&(x["steam_price"].isna())).sum()
    trade_day_missing_ffill=((x["steam_valid"]==1)&(x["steam_price_ffill"].isna())).sum()

    first_trade_idx=x.index[x["steam_valid"]==1]
    fill_should_exist_missing=0
    if len(first_trade_idx)>0:
        first_trade_i=int(first_trade_idx[0])
        y=x.iloc[first_trade_i:].copy()
        fill_should_exist_missing=((y["steam_valid"]==0)&(y["days_since_last_trade"]<=ffill_limit_days)&(y["steam_price_ffill"].isna())).sum()

    missing_ffill_total=int(x["steam_price_ffill"].isna().sum())
    total_rows=int(len(x))

    ok=(int(missing_gaps)==0 and int(vol_bad)==0 and int(fill_over_limit)==0 and int(trade_day_missing_price)==0 and int(trade_day_missing_ffill)==0 and int(fill_should_exist_missing)==0)

    print(key)
    print("missing_day_gaps_gt1:",int(missing_gaps))
    print("volume_nonzero_on_no_trade_days:",int(vol_bad))
    print("ffill_over_limit:",int(fill_over_limit))
    print("trade_day_missing_price:",int(trade_day_missing_price))
    print("trade_day_missing_ffill:",int(trade_day_missing_ffill))
    print("ffill_missing_when_gap_le_limit:",int(fill_should_exist_missing))
    print("missing_ffill_total:",missing_ffill_total,"/",total_rows)
    print("status:","ok" if ok else "issues")
    print("-"*40)

    return ok

def rebased_index(panel):
    x=panel[["date","steam_price_ffill"]].copy()
    x=x.dropna()
    if x.empty:
        return None
    base=float(x["steam_price_ffill"].iloc[0])
    if base<=0:
        return None
    x["index"]=100.0*(x["steam_price_ffill"].astype(float)/base)
    return x[["date","index"]]

def daily_returns(panel):
    x=panel[["date","steam_price_ffill","steam_volume","steam_valid"]].copy()
    x=x.dropna(subset=["steam_price_ffill"])
    x["p"]=x["steam_price_ffill"].astype(float)
    x["ret"]=x["p"].pct_change()
    return x

def plot_missing_timeline(panel,title):
    x=panel.copy()
    miss=x["missing_raw"].astype(int)

    plt.figure()
    plt.plot(x["date"],miss)
    plt.title(title+" | missing raw price (1=missing)")
    plt.tight_layout()
    plt.show()

def plot_gap_hist(panel,title):
    x=panel.copy()
    y=x["days_since_last_trade"].copy()
    y=y[(y>=0)&(y<200)]
    if len(y)<10:
        return

    plt.figure()
    plt.hist(y,bins=60)
    plt.title(title+" | gap length distribution (capped at 200 days)")
    plt.tight_layout()
    plt.show()

def plot_missing_before_after(panel,title):
    x=panel.copy()
    m_raw=float(x["missing_raw"].mean())
    m_ff=float(x["missing_ffill"].mean())

    plt.figure()
    plt.bar(("raw_missing","ffill_missing"),(m_raw,m_ff))
    plt.title(title+" | missing ratio before vs after ffill")
    plt.tight_layout()
    plt.show()

def plot_missing_heatmap(panels):
    rows=list()
    for k,p in panels.items():
        x=p[["date","missing_raw"]].copy()
        x=x.rename(columns=dict(missing_raw=k))
        rows.append(x)

    if not rows:
        return

    m=rows[0]
    for r in rows[1:]:
        m=m.merge(r,how="outer",on="date")

    m=m.sort_values("date").fillna(1).astype({c:int for c in m.columns if c!="date"})

    cols=[c for c in m.columns if c!="date"]
    mat=m[cols].T.values

    plt.figure()
    plt.imshow(mat,aspect="auto",interpolation="nearest")
    plt.yticks(range(len(cols)),cols)
    plt.title("missingness heatmap (raw price) | 1=missing")
    plt.tight_layout()
    plt.show()

def plot_item_series(panel,title):
    plt.figure()
    plt.plot(panel["date"],panel["steam_price"],label="steam_price")
    plt.plot(panel["date"],panel["steam_price_ffill"],label="steam_price_ffill")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.plot(panel["date"],panel["steam_volume"])
    plt.title(title+" | volume")
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.plot(panel["date"],panel["days_since_last_trade"])
    plt.title(title+" | days_since_last_trade")
    plt.tight_layout()
    plt.show()

    plot_missing_timeline(panel,title)
    plot_gap_hist(panel,title)
    plot_missing_before_after(panel,title)

def plot_liquidity_hist(panels):
    rows=list()
    for k,p in panels.items():
        active=float(p["steam_valid"].mean())
        avg_vol=float(p["steam_volume"].mean())
        fill_rate=float(p["price_was_filled"].mean())
        rows.append((k,active,avg_vol,fill_rate))
    df=pd.DataFrame(rows,columns=("key","active_rate","avg_vol","fill_rate"))

    plt.figure()
    plt.hist(df["active_rate"],bins=12)
    plt.title("active_rate distribution (selected items)")
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.hist(df["avg_vol"],bins=12)
    plt.title("avg daily volume distribution (selected items)")
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.hist(df["fill_rate"],bins=12)
    plt.title("fill_rate distribution (selected items)")
    plt.tight_layout()
    plt.show()

def plot_rebased(indices):
    m=None
    for k,df in indices.items():
        if df is None or df.empty:
            continue
        tmp=df.rename(columns=dict(index=k))
        if m is None:
            m=tmp
        else:
            m=m.merge(tmp,how="outer",on="date")
    if m is None:
        return None

    m=m.sort_values("date")

    plt.figure()
    for col in [c for c in m.columns if c!="date"]:
        plt.plot(m["date"],m[col],label=col)
    plt.title("rebased indices (start=100)")
    plt.legend()
    plt.tight_layout()
    plt.show()
    return m

def plot_rolling_corr(m,xcol,ycol,window):
    z=m[["date",xcol,ycol]].dropna().copy()
    z["roll_corr"]=z[xcol].rolling(window).corr(z[ycol])

    plt.figure()
    plt.plot(z["date"],z["roll_corr"])
    plt.title("rolling correlation: "+xcol+" vs "+ycol+" (window="+str(window)+")")
    plt.tight_layout()
    plt.show()

def plot_corr_heatmap(m):
    cols=[c for c in m.columns if c!="date"]
    z=m[cols].copy()
    corr=z.corr()

    plt.figure()
    plt.imshow(corr.values,interpolation="nearest")
    plt.xticks(range(len(cols)),cols,rotation=30,ha="right")
    plt.yticks(range(len(cols)),cols)
    plt.title("correlation heatmap (rebased indices)")
    plt.tight_layout()
    plt.show()

def plot_scatter_price_vs_volume(panels):
    for k,p in panels.items():
        x=p[["steam_price_ffill","steam_volume","steam_valid"]].copy()
        x=x.dropna(subset=["steam_price_ffill"])
        x=x[x["steam_valid"]==1]
        if len(x)<200:
            continue

        plt.figure()
        plt.scatter(x["steam_volume"],x["steam_price_ffill"].astype(float))
        plt.title(k+" | scatter: volume vs price (trade days)")
        plt.xlabel("volume")
        plt.ylabel("price")
        plt.tight_layout()
        plt.show()

def plot_scatter_returns_vs_volume(panels):
    for k,p in panels.items():
        r=daily_returns(p).dropna(subset=["ret"])
        r=r[r["steam_valid"]==1]
        if len(r)<200:
            continue

        plt.figure()
        plt.scatter(r["steam_volume"],r["ret"])
        plt.title(k+" | scatter: volume vs return (trade days)")
        plt.xlabel("volume")
        plt.ylabel("daily return")
        plt.tight_layout()
        plt.show()

def plot_scatter_liquidity_vs_volatility(panels):
    rows=list()
    for k,p in panels.items():
        r=daily_returns(p).dropna(subset=["ret"])
        if len(r)<80:
            continue
        vol=float(r["ret"].rolling(roll_v).std().median())
        active=float(p["steam_valid"].mean())
        avg_vol=float(p["steam_volume"].mean())
        rows.append((k,active,avg_vol,vol))
    df=pd.DataFrame(rows,columns=("key","active_rate","avg_vol","volatility"))

    if df.empty:
        return

    plt.figure()
    plt.scatter(df["active_rate"],df["volatility"])
    plt.title("scatter: active_rate vs volatility (30d vol median)")
    plt.xlabel("active_rate")
    plt.ylabel("volatility")
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.scatter(df["avg_vol"],df["volatility"])
    plt.title("scatter: avg_vol vs volatility (30d vol median)")
    plt.xlabel("avg_vol")
    plt.ylabel("volatility")
    plt.tight_layout()
    plt.show()

def plot_box_returns(panels):
    data=list()
    labels=list()
    for k,p in panels.items():
        r=daily_returns(p).dropna(subset=["ret"])
        if len(r)<120:
            continue
        data.append(r["ret"].dropna().values)
        labels.append(k)

    if not data:
        return

    plt.figure()
    plt.boxplot(data,tick_labels=labels,showfliers=False)
    plt.title("daily returns distribution (boxplot)")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()

def main():
    conn=psycopg2.connect(db_conn)

    global_min,global_max=get_global_date_range(conn)
    print("Global data range:",global_min,"–",global_max)

    resolved=dict()
    for key,base in items:
        name=pick_best_match(conn,base)
        resolved[key]=name

    print("resolved")
    for k,v in resolved.items():
        print(k,":",v)
    print("-"*40)

    panels=dict()
    summaries=dict()
    for k,item_name in resolved.items():
        if not item_name:
            continue
        daily=fetch_all_available(conn,item_name)
        if daily.empty:
            continue
        panel,summ=build_daily_panel(daily,ffill_limit)
        panels[k]=panel
        summaries[k]=summ

    conn.close()

    for k,p in panels.items():
        sanity(p,k,ffill_limit)

    print("\nEnhanced data quality summary:")
    print(f"{'Key':<12} {'Total days':>12} {'Raw missing':>12} {'Raw %':>10} {'After ffill':>12} {'After %':>10}")
    print("-"*66)
    for k in sorted(summaries.keys()):
        s=summaries[k]
        print(f"{k:<12} {s['total_days']:12d} {s['raw_missing']:12d} {s['raw_missing_pct']:10.2f}% {s['ffill_missing']:12d} {s['ffill_missing_pct']:10.2f}%")

    plot_missing_heatmap(panels)

    for k,p in panels.items():
        if k in ("knife","ak_red","deagle_red","case_common","case_new","butterfly_ch_fn"):
            plot_item_series(p,resolved.get(k,k))

    plot_liquidity_hist(panels)

    indices=dict()
    for k,p in panels.items():
        indices[k]=rebased_index(p)

    m=plot_rebased(indices)

    if m is not None:
        if "knife" in m.columns and "ak_red" in m.columns:
            plot_rolling_corr(m,"knife","ak_red",roll_c)
        if "knife" in m.columns and "deagle_red" in m.columns:
            plot_rolling_corr(m,"knife","deagle_red",roll_c)

        if "knife" in m.columns and "butterfly_ch_fn" in m.columns:
            plot_rolling_corr(m,"knife","butterfly_ch_fn",roll_c)
        plot_corr_heatmap(m)

    plot_scatter_price_vs_volume(panels)
    plot_scatter_returns_vs_volume(panels)
    plot_scatter_liquidity_vs_volatility(panels)
    plot_box_returns(panels)

if __name__=="__main__":
    main()