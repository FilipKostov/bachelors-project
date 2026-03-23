from pathlib import Path
import numpy as np
import pandas as pd

out_path=Path("F:/steam_market_project/data/model_training_out")
VARIANTS=["raw","raw_liquipedia","raw_fe_liquipedia"]

DATE_COL="date"
ITEM_COL="item_name"
IDX_COL_PREFIX="idx_"
metric="mae"

def rmse(y_true,y_pred):
    y_true=np.asarray(y_true,dtype=float)
    y_pred=np.asarray(y_pred,dtype=float)
    m=np.isfinite(y_true)&np.isfinite(y_pred)
    if m.sum()==0:return float("nan")
    return float(np.sqrt(np.mean((y_true[m]-y_pred[m])**2)))

def mae(y_true,y_pred):
    y_true=np.asarray(y_true,dtype=float)
    y_pred=np.asarray(y_pred,dtype=float)
    m=np.isfinite(y_true)&np.isfinite(y_pred)
    if m.sum()==0:return float("nan")
    return float(np.mean(np.abs(y_true[m]-y_pred[m])))

def mape(y_true,y_pred):
    y_true=np.asarray(y_true,dtype=float)
    y_pred=np.asarray(y_pred,dtype=float)
    m=np.isfinite(y_true)&np.isfinite(y_pred)&(np.abs(y_true)>1e-12)
    if m.sum()==0:return float("nan")
    return float(np.mean(np.abs((y_true[m]-y_pred[m])/y_true[m])))

def smape(y_true,y_pred):
    y_true=np.asarray(y_true,dtype=float)
    y_pred=np.asarray(y_pred,dtype=float)
    m=np.isfinite(y_true)&np.isfinite(y_pred)
    if m.sum()==0:return float("nan")
    denom=np.abs(y_true[m])+np.abs(y_pred[m])
    denom=np.where(denom<1e-12,1e-12,denom)
    return float(np.mean(2.0*np.abs(y_pred[m]-y_true[m])/denom))

def wape(y_true,y_pred):
    y_true=np.asarray(y_true,dtype=float)
    y_pred=np.asarray(y_pred,dtype=float)
    m=np.isfinite(y_true)&np.isfinite(y_pred)
    if m.sum()==0:return float("nan")
    denom=np.sum(np.abs(y_true[m]))
    if denom<1e-12:return float("nan")
    return float(np.sum(np.abs(y_true[m]-y_pred[m]))/denom)

def directional_acc(y_true,y_pred):
    y_true=pd.Series(y_true).astype(float).reset_index(drop=True)
    y_pred=pd.Series(y_pred).astype(float).reset_index(drop=True)
    dy_true=y_true.diff()
    dy_pred=y_pred.diff()
    m=dy_true.notna()&dy_pred.notna()
    if m.sum()==0:return float("nan")
    return float((np.sign(dy_true[m])==np.sign(dy_pred[m])).mean())

def metric_row(df):
    y_true=df["y_true"].astype(float).values
    y_pred=df["y_pred"].astype(float).values
    return {"mae":mae(y_true,y_pred),"rmse":rmse(y_true,y_pred),"mape":mape(y_true,y_pred),"smape":smape(y_true,y_pred),"wape":wape(y_true,y_pred),"directional_acc":directional_acc(y_true,y_pred),"n":int(len(df))}

def split_key(name):
    if name.endswith("_test"):return "test"
    if name.endswith("_val"):return "val"
    return ""

def analyze_one_variant(out_dir):
    pred_files=sorted(Path(out_dir).glob("preds_*.csv"))
    if not pred_files:raise RuntimeError("no preds_*.csv in "+str(out_dir))
    rows=list()
    for f in pred_files:
        name=f.stem.replace("preds_","")
        df=pd.read_csv(f)
        rows.append({"name":name,**metric_row(df)})
    all_res=pd.DataFrame(rows)
    base_map={}
    for _,r in all_res.iterrows():
        nm=str(r["name"])
        if nm.startswith("baseline1_"):base_map[("baseline1",split_key(nm))]=r
        if nm.startswith("baseline2_movavg_"):base_map[("baseline2",split_key(nm))]=r
    out_rows=list()
    for _,r in all_res.iterrows():
        nm=str(r["name"])
        sk=split_key(nm)
        row={"split":sk,"name":nm,"mae":float(r["mae"]),"rmse":float(r["rmse"]),"mape":float(r["mape"]),"smape":float(r["smape"]),"wape":float(r["wape"]),"directional_acc":float(r["directional_acc"]),"n":int(r["n"])}
        b1=base_map.get(("baseline1", sk))
        if b1 is not None and not nm.startswith("baseline1_"):
            b1_mae=float(b1["mae"])
            b1_rmse=float(b1["rmse"])
            row["baseline1_mae"]=b1_mae
            row["baseline1_rmse"]=b1_rmse
            row["mae_improve_vs_baseline1"]=(b1_mae - row["mae"]) / b1_mae if b1_mae > 0 else float("nan")
            row["rmse_improve_vs_baseline1"]=(b1_rmse - row["rmse"]) / b1_rmse if b1_rmse > 0 else float("nan")
        else:
            row["baseline1_mae"]=float("nan")
            row["baseline1_rmse"]=float("nan")
            row["mae_improve_vs_baseline1"]=float("nan")
            row["rmse_improve_vs_baseline1"]=float("nan")
        b2=base_map.get(("baseline2", sk))
        if b2 is not None and not nm.startswith("baseline"):
            b2_mae=float(b2["mae"])
            b2_rmse=float(b2["rmse"])
            row["baseline2_mae"]=b2_mae
            row["baseline2_rmse"]=b2_rmse
            row["mae_improve_vs_baseline2"]=float((b2_mae - row["mae"]) / b2_mae) if b2_mae > 0 else float("nan")
            row["rmse_improve_vs_baseline2"]=float((b2_rmse - row["rmse"]) / b2_rmse) if b2_rmse > 0 else float("nan")
        else:
            row["baseline2_mae"]=float("nan")
            row["baseline2_rmse"]=float("nan")
            row["mae_improve_vs_baseline2"]=float("nan")
            row["rmse_improve_vs_baseline2"]=float("nan")
        out_rows.append(row)
    res=pd.DataFrame(out_rows).sort_values(["split","name"]).reset_index(drop=True)
    res.to_csv(Path(out_dir)/"results_table.csv",index=False)
    per_item_dir=Path(out_dir)/"per_item"
    per_item_dir.mkdir(parents=True,exist_ok=True)
    for f in pred_files:
        nm=f.stem.replace("preds_","")
        df=pd.read_csv(f)
        g=df.groupby(ITEM_COL,dropna=False,sort=False)
        out=[]
        for item,sub in g:
            r=metric_row(sub)
            r[ITEM_COL]=item
            out.append(r)
        out_df=pd.DataFrame(out).sort_values("mae",ascending=True).reset_index(drop=True)
        out_df.to_csv(per_item_dir/f"per_item_{nm}.csv",index=False)
        best=out_df.head(30)
        worst=out_df.tail(30).sort_values("mae",ascending=False)
        best.to_csv(per_item_dir/f"best30_{nm}.csv",index=False)
        worst.to_csv(per_item_dir/f"worst30_{nm}.csv",index=False)
    cat_out=list()
    for f in pred_files:
        nm=f.stem.replace("preds_","")
        sk=split_key(nm)
        if sk not in ["val","test"]:continue
        meta_path=Path(out_dir)/f"meta_{sk}.csv"
        if not meta_path.exists():continue
        df=pd.read_csv(f)
        meta=pd.read_csv(meta_path)
        m=df.merge(meta,on=[DATE_COL,ITEM_COL],how="left")
        idx_cols=[c for c in m.columns if c.startswith(IDX_COL_PREFIX)]
        if not idx_cols:continue
        for idx in idx_cols:
            sub=m[m[idx].fillna(0).astype(int)==1].copy()
            if len(sub)<200:continue
            r=metric_row(sub)
            r["name"]=nm
            r["split"]=sk
            r["category"]=idx
            cat_out.append(r)
    if cat_out:
        cat_df=pd.DataFrame(cat_out).sort_values(["split","name","mae"]).reset_index(drop=True)
        cat_df.to_csv(Path(out_dir)/"category_results.csv",index=False)
    return res

def extract_metric_table(res_df,variant,split_name,metric_name):
    sub=res_df[res_df["split"]==split_name].copy()
    if sub.empty:return pd.DataFrame(columns=["variant","split","name",metric_name])
    out=sub[["name",metric_name]].copy()
    out["variant"]=variant
    out["split"]=split_name
    return out[["variant","split","name",metric_name]]

def compare_variants(all_tables,split_name,metric_name,out_path):
    wide=None
    for t in all_tables:
        v=str(t["variant"])
        df=t["df"]
        if df is None or df.empty:continue
        a=extract_metric_table(df,v,split_name,metric_name)
        if a.empty:continue
        if wide is None:
            wide=a[["name",metric_name]].rename(columns={metric_name:v})
        else:
            wide=wide.merge(a[["name",metric_name]].rename(columns={metric_name:v}),on="name",how="outer")
    if wide is None:return
    for v in ["raw","raw_liquipedia","raw_fe_liquipedia"]:
        if v not in wide.columns:wide[v]=np.nan
    wide["diff_liquipedia"]=wide["raw_liquipedia"]-wide["raw"]
    wide["diff_fe_liquipedia"]=wide["raw_fe_liquipedia"]-wide["raw"]
    denom=wide["raw"].astype(float).replace([0.0,-0.0],np.nan)
    wide["pct_liquipedia"]=wide["diff_liquipedia"]/denom
    wide["pct_fe_liquipedia"]=wide["diff_fe_liquipedia"]/denom
    wide=wide.sort_values("diff_fe_liquipedia",ascending=True).reset_index(drop=True)
    wide.to_csv(out_path,index=False)

def main():
    all_tables=list()
    for v in VARIANTS:
        out_dir=out_path/v
        if not out_dir.exists():
            all_tables.append({"variant":v,"df":pd.DataFrame()})
            continue
        res=analyze_one_variant(out_dir)
        all_tables.append({"variant":v,"df":res})
    compare_variants(all_tables,"val",metric,out_path/f"compare_variants_val_{metric}.csv")
    compare_variants(all_tables,"test",metric,out_path/f"compare_variants_test_{metric}.csv")

if __name__=="__main__":
    main()