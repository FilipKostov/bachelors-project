import os,json,math,glob
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error,mean_squared_error
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,TensorDataset
import copy

INPUT_VARIANTS=[("raw",Path("F:/steam_market_project/data/model_subset_csv")),("raw_liquipedia",Path("F:/steam_market_project/data/model_subset_csv_raw_liquipedia")),("raw_fe_liquipedia",Path("F:/steam_market_project/data/model_subset_csv_raw_fe_liquipedia")),]

OUT_ROOT=Path("F:/steam_market_project/data/model_training_out")
OUT_ROOT.mkdir(parents=True,exist_ok=True)

TARGET="y_avg_7d_fwd"
DATE="date"
ITEM="item_name"

TR_PCT=0.70
VA_PCT=0.15
TE_PCT=0.15

MAX_FILES=None
SEED=42
MAX_PLOT=4000

BASE_HINTS=["steam_price_ffill7","steam_price","price_ffill","price","avg_price"]
MA_HINTS=["roll_mean_7","roll_mean7","ma_7","movavg_7"]

LOG_TARGET=True
ADD_NA_FLAGS=True
TUNE_XGB=True
TUNE_TRIALS=30
TUNE_METRIC="mae"

PRICE_CAP=1800.0
CAP_THR=0.84
CAP_VOL_FRAC=0.20
MIN_CAP_VOL=25
CAP_FILL=-2000.0

def get_files(p):
    fs=sorted(glob.glob(str(Path(p)/"subset_*.csv")))
    if MAX_FILES:fs=fs[:MAX_FILES]
    if not fs:
        raise RuntimeError("no files")
    return fs

def rmse(y,yh):
    return float(math.sqrt(mean_squared_error(y,yh)))

def mape(y,yh):
    y,yh=np.asarray(y,dtype=float),np.asarray(yh,dtype=float)
    m=np.isfinite(y)&np.isfinite(yh)&(np.abs(y)>1e-12)
    if not m.sum():
        return float("nan")
    return float(np.mean(np.abs((y[m]-yh[m])/y[m])))

def smape(y,yh):
    y,yh=np.asarray(y,dtype=float),np.asarray(yh,dtype=float)
    m=np.isfinite(y)&np.isfinite(yh)
    if not m.sum():
        return float("nan")
    d=np.abs(y[m])+np.abs(yh[m])
    d=np.where(d<1e-12,1e-12,d)
    return float(np.mean(2*np.abs(yh[m]-y[m])/d))

def wape(y,yh):
    y,yh=np.asarray(y,dtype=float),np.asarray(yh,dtype=float)
    m=np.isfinite(y)&np.isfinite(yh)
    if not m.sum():
        return float("nan")
    d=float(np.sum(np.abs(y[m])))
    if d<1e-12:
        return float("nan")
    return float(np.sum(np.abs(y[m]-yh[m]))/d)

def dir_acc(y,yh):
    dy,dyh=pd.Series(y).diff(),pd.Series(yh).diff()
    m=dy.notna()&dyh.notna()
    if not m.sum():
        return float("nan")
    return float((np.sign(dy[m])==np.sign(dyh[m])).mean())

def read_all(fs):
    return pd.concat([pd.read_csv(f)for f in fs],ignore_index=True)

def split_time(df):
    df=df.copy()
    df[DATE]=pd.to_datetime(df[DATE],errors="coerce")
    df=df.dropna(subset=[DATE,TARGET])
    df[TARGET]=pd.to_numeric(df[TARGET],errors="coerce")
    df=df.replace([np.inf,-np.inf],np.nan).sort_values([DATE,ITEM]).reset_index(drop=True)
    mn,mx=df[DATE].min(),df[DATE].max()
    days=(mx-mn).days
    tr_end=mn+pd.Timedelta(days=int(days*TR_PCT))
    va_end=mn+pd.Timedelta(days=int(days*(TR_PCT+VA_PCT)))
    return (df[df[DATE]<=tr_end].copy(),df[(df[DATE]>tr_end)&(df[DATE]<=va_end)].copy(),df[df[DATE]>va_end].copy())

def item_map(df):
    return {n:i for i,n in enumerate(sorted(df[ITEM].astype(str).unique()))}

def add_id(df,m):
    df=df.copy()
    df[ITEM]=df[ITEM].astype(str)
    df["item_id"]=df[ITEM].map(m).fillna(-1).astype("int32")
    return df

def num_cols(df):
    return [c for c in df.columns if c not in {TARGET,ITEM,DATE}]

def to_num(df,cols):
    df=df.copy()
    for c in cols:df[c]=pd.to_numeric(df[c],errors="coerce")
    return df

def metrics(y,yh):
    return {"mae":float(mean_absolute_error(y,yh)),"rmse":rmse(y,yh),"mape":mape(y,yh),"smape":smape(y,yh),"wape":wape(y,yh),"dir_acc":dir_acc(y,yh)}

def find_hint_col(cands,hints,avoid=[]):
    lc={c.lower():c for c in cands}
    for h in hints:
        hl=h.lower()
        for l,o in lc.items():
            if hl==l or hl in l:
                if not any(a in l for a in avoid):
                    return o
    return None

def find_price(df,feats):
    return find_hint_col(feats,BASE_HINTS,["fwd"])or next((c for c in feats if"price"in c.lower()and"fwd"not in c.lower()),None)

def find_ma(df,feats):
    return find_hint_col(feats,MA_HINTS,["fwd"])

def add_na_flags(df,cols):
    return pd.concat([df,pd.DataFrame({f"isna_{c}":df[c].isna().astype("int8")for c in cols},index=df.index)],axis=1)

def ffill_limit(df,cols,maxg=7):
    df=df.sort_values([ITEM,DATE]).reset_index(drop=True)
    for c in cols:df[c]=df.groupby(ITEM)[c].transform(lambda x:x.ffill(limit=maxg))
    return df

def cap_items(df,pcol="steam_price_ffill7",vcol="volume"):
    df=df.copy()
    df["_nc"]=(df[pcol]>=CAP_THR*PRICE_CAP).astype(int)
    df["_vn"]=df[vcol]*df["_nc"]
    agg=df.groupby(ITEM).agg(tv=(vcol,"sum"),nv=("_vn","sum"),nc=("_nc","sum"),hpd=(pcol,lambda x:((x>=1200)&x.notna()).sum()),td=("date","nunique")).reset_index()
    agg["fnc"]=agg["nv"]/agg["tv"].replace(0,np.nan)
    agg["cap"]=( (agg["fnc"]>=CAP_VOL_FRAC)&(agg["nc"]>=MIN_CAP_VOL) | (agg["hpd"]/agg["td"].replace(0,1)>=0.6) ).astype("int8")
    df=df.merge(agg[[ITEM,"cap"]],on=ITEM,how="left")
    df["near_cap"]=df["_nc"].astype("int8")
    df.drop(columns=["_nc","_vn"],errors="ignore",inplace=True)
    return df

def cap_fill(df,pcol="steam_price_ffill7"):
    df=df.copy().sort_values([ITEM,DATE]).reset_index(drop=True)
    df["pp"]=df.groupby(ITEM)[pcol].shift(1)
    mg=(df[pcol].isna())&(df["cap"]==1)&(df["pp"]>=CAP_THR*PRICE_CAP)
    df["cc"]=df.groupby(ITEM).cumcount()
    df["lv"]=df.groupby(ITEM)[pcol].transform(lambda x:x.notna().cumsum().where(x.notna()))
    df["gd"]=df["cc"]-df.groupby(ITEM)["lv"].ffill()
    df.loc[mg&(df["gd"]<=90),pcol]=df.loc[mg&(df["gd"]<=90),"pp"]
    df.loc[mg&(df["gd"]>90),pcol]=CAP_FILL
    df.drop(columns=["pp","cc","lv","gd"],errors="ignore",inplace=True)
    return df

def fit_xgb(Xt,yt,Xv,yv,p):
    from xgboost import XGBRegressor
    m=XGBRegressor(**p,early_stopping_rounds=60,eval_metric="mae")
    m.fit(Xt,yt,eval_set=[(Xv,yv)],verbose=False)
    print(f"best it {m.best_iteration}")
    return m

def tune_xgb(Xt,yt,Xv,yv,s):
    rs=np.random.RandomState(s)
    best_s,best_p,best_m=None,None,None
    base={"random_state":SEED,"n_jobs":max(1,os.cpu_count()-1),"tree_method":"hist","objective":"reg:squarederror"}
    for i in range(TUNE_TRIALS):
        p=dict(base)
        p["n_estimators"]=int(rs.choice([600,800,1000,1200,1500]))
        p["learning_rate"]=float(rs.choice([0.02,0.03,0.05,0.08]))
        p["max_depth"]=int(rs.choice([5,6,7,8,10]))
        p["subsample"]=float(rs.choice([0.7,0.8,0.85,0.9,1.0]))
        p["colsample_bytree"]=float(rs.choice([0.7,0.8,0.85,0.9,1.0]))
        p["reg_lambda"]=float(rs.choice([0.5,1.0,2.0,4.0,8.0]))
        p["reg_alpha"]=float(rs.choice([0.0,0.01,0.05,0.1]))
        p["min_child_weight"]=float(rs.choice([1.0,2.0,5.0,10.0]))
        p["gamma"]=float(rs.choice([0.0,0.1,0.2]))
        m=fit_xgb(Xt,yt,Xv,yv,p)
        pr=m.predict(Xv)
        sc=mean_absolute_error(yv,pr)if TUNE_METRIC=="mae"else rmse(yv,pr)
        if best_s is None or sc<best_s:
            best_s,best_p,best_m=sc,p,m
        print(f"[{s}] {i+1}/{TUNE_TRIALS} {TUNE_METRIC}={sc:.6f} best={best_s:.6f if best_s else 'none'}")
    return best_m,best_p,best_s

class MLP(nn.Module):
    def __init__(self,n_items,ed=48,idim=None):
        super().__init__()
        self.e=nn.Embedding(n_items,ed)
        self.head=nn.Sequential(nn.Linear(ed+idim,512),nn.ReLU(),nn.Dropout(0.1),nn.Linear(512,256),nn.ReLU(),nn.Dropout(0.1),nn.Linear(256,128),nn.ReLU(),nn.Linear(128,64),nn.ReLU(),nn.Linear(64,1))
    def forward(self,xo,ii):
        return self.head(torch.cat([self.e(ii),xo],1))

def train_mlp(Xi,y,feats,imap,tdf,dev=None):
    if dev is None:dev=torch.device("cuda"if torch.cuda.is_available()else"cpu")
    print(f"mlp on {dev}")
    n_items=len(imap)
    nums=[c for c in feats if not c.startswith("isna_")]
    nidx=[feats.index(c)for c in nums]
    sc=StandardScaler().fit(Xi[:,nidx])
    def prep(X,df):
        ii=df[ITEM].map(imap).values.astype(np.int64)
        xn=sc.transform(X[:,nidx]).astype(np.float32)
        return torch.from_numpy(xn).to(dev),torch.from_numpy(ii).long().to(dev)
    Xo,ii=prep(Xi,tdf)
    yt=torch.from_numpy(y.astype(np.float32)).unsqueeze(1).to(dev)
    dl=DataLoader(TensorDataset(Xo,ii,yt),4096,shuffle=True)
    m=MLP(n_items,input_dim=Xo.shape[1]).to(dev)
    opt=optim.AdamW(m.parameters(),lr=0.001,weight_decay=1e-4)
    lossf=nn.MSELoss()
    bl,bs,pc=float("inf"),None,0
    for ep in range(500):
        m.train()
        tl=0
        for xo,ii,yb in dl:
            opt.zero_grad()
            p=m(xo,ii)
            l=lossf(p,yb)
            l.backward()
            opt.step()
            tl+=l.item()*yb.size(0)
        al=tl/len(dl.dataset)
        print(f"ep {ep+1:3d} mse {al:.6f}")
        if al<bl:
            bl=al
            bs=copy.deepcopy(m.state_dict())
            pc=0
        else:
            pc+=1
            if pc>=40:
                print(f"stop ep {ep+1}")
                break
    if bs:
        m.load_state_dict(bs)
    m.eval()
    return m,sc,prep

def run(tag,indir,outdir):
    outdir=Path(outdir)
    outdir.mkdir(parents=True,exist_ok=True)
    np.random.seed(SEED)
    print(f"\n{tag} → {outdir.name}")
    fs=get_files(indir)
    df=read_all(fs)
    print(f"{len(df):,} rows")
    df=cap_items(df)
    df=cap_fill(df)
    tr,va,te=split_time(df)
    all_d=pd.concat([tr,va,te],ignore_index=True)
    imap=item_map(all_d)
    tr=add_id(tr,imap)
    va=add_id(va,imap)
    te=add_id(te,imap)
    feats=num_cols(tr)
    if "cap"not in feats:
        feats.append("cap")
    xfeats=[c for c in feats if c!="item_id"]
    mfeats=xfeats[:]
    tr=to_num(tr,feats)
    va=to_num(va,feats)
    te=to_num(te,feats)
    basec=find_price(tr,feats)
    mac=find_ma(tr,feats)
    if ADD_NA_FLAGS:
        of=feats[:]
        tr=add_na_flags(tr,of)
        va=add_na_flags(va,of)
        te=add_na_flags(te,of)
        feats=sorted(set(feats+[f"isna_{c}"for c in of]))
        xfeats=[c for c in feats if c!="item_id"]
        mfeats=xfeats[:]
    ffcols=[c for c in feats if"volume"not in c.lower()and c!=TARGET]
    tr=ffill_limit(tr,ffcols)
    va=ffill_limit(va,ffcols)
    te=ffill_limit(te,ffcols)
    imp=SimpleImputer(strategy="median")
    Xt=imp.fit_transform(tr[xfeats])
    Xv=imp.transform(va[xfeats])
    Xte=imp.transform(te[xfeats])
    yt=tr[TARGET].astype(float).values
    yv=va[TARGET].astype(float).values
    yte=te[TARGET].astype(float).values
    print("\nxgb raw")
    if TUNE_XGB:
        xgbr,praw,_=tune_xgb(Xt,yt,Xv,yv,SEED+1)
        (outdir/"xgb_raw_params.json").write_text(json.dumps(praw,indent=2))
    else:
        praw={"n_estimators":1200,"learning_rate":0.03,"max_depth":8,"subsample":0.85,"colsample_bytree":0.85,"reg_lambda":2.0,"reg_alpha":0.0,"min_child_weight":1.0,"gamma":0.0,"random_state":SEED,"n_jobs":max(1,os.cpu_count()-1),"tree_method":"hist","objective":"reg:squarederror"}
        xgbr=fit_xgb(Xt,yt,Xv,yv,praw)
    pv_xr=xgbr.predict(Xv)
    pt_xr=xgbr.predict(Xte)
    print("\nmlp raw")
    mlpr,scr,prepr=train_mlp(Xt,yt,mfeats,imap,tr)
    with torch.no_grad():
        Xvo,ivo=prepr(Xv,va)
        pv_mr=mlpr(Xvo,ivo).cpu().numpy().flatten()
        Xteo,ite=prepr(Xte,te)
        pt_mr=mlpr(Xteo,ite).cpu().numpy().flatten()
    if LOG_TARGET:
        ll=np.log1p(1e-5)
        lh=9.0
        ytl=np.log1p(np.maximum(yt,0))
        yvl=np.log1p(np.maximum(yv,0))
        print("\nxgb log")
        if TUNE_XGB:
            xbgl,plog,_=tune_xgb(Xt,ytl,Xv,yvl,SEED+2)
            (outdir/"xgb_log_params.json").write_text(json.dumps(plog,indent=2))
        else:xbgl=fit_xgb(Xt,ytl,Xv,yvl,praw)
        pv_xl=np.expm1(np.clip(xbgl.predict(Xv),ll,lh))
        pt_xl=np.expm1(np.clip(xbgl.predict(Xte),ll,lh))
        pv_xl=np.maximum(pv_xl,0)
        pt_xl=np.maximum(pt_xl,0)
        print("\nmlp log")
        mlpl,scl,prepl=train_mlp(Xt,ytl,mfeats,imap,tr)
        with torch.no_grad():
            Xvol,ivol=prepl(Xv,va)
            lv=mlpl(Xvol,ivol).cpu().numpy().flatten()
            Xteol,itel=prepl(Xte,te)
            lt=mlpl(Xteol,itel).cpu().numpy().flatten()
        pv_ml=np.maximum(np.expm1(np.clip(lv,ll,lh)),0)
        pt_ml=np.maximum(np.expm1(np.clip(lt,ll,lh)),0)
        ev=(pv_xr+pv_xl+pv_mr+pv_ml)/4
        et=(pt_xr+pt_xl+pt_mr+pt_ml)/4
    else:
        ev=(pv_xr+pv_mr)/2
        et=(pt_xr+pt_mr)/2
    print("\nval")
    print(json.dumps(metrics(yv,ev),indent=2))
    print("\ntest")
    print(json.dumps(metrics(yte,et),indent=2))
    def save(df,y,yh,nm):
        o=df[[DATE,ITEM]].copy()
        o["true"]=y
        o["pred"]=yh
        o.sort_values([DATE,ITEM]).to_csv(outdir/f"pred_{nm}.csv",index=False)
    def plot(y,yh,t,png):
        n=min(len(y),MAX_PLOT)
        plt.figure(figsize=(10,5))
        plt.plot(y[:n],label="true")
        plt.plot(yh[:n],label="pred")
        plt.title(t)
        plt.legend()
        plt.tight_layout()
        plt.savefig(png,dpi=160)
        plt.close()
    save(va,yv,ev,"val")
    save(te,yte,et,"test")
    plot(yte,et,f"{tag} test",outdir/"test.png")
    plot(yv,ev,f"{tag} val",outdir/"val.png")
    joblib.dump({"imp":imp,"m":xgbr,"feats":xfeats,"base":basec,"ma":mac},outdir/"xgb_raw.jlb")
    torch.save(mlpr.state_dict(),outdir/"mlp_raw.pth")
    joblib.dump({"imp":imp,"sc":scr,"feats":mfeats,"imap":imap},outdir/"meta_raw.jlb")
    if LOG_TARGET:
        joblib.dump({"imp":imp,"m":xbgl,"feats":xfeats,"base":basec,"ma":mac},outdir/"xgb_log.jlb")
        torch.save(mlpl.state_dict(),outdir/"mlp_log.pth")
        joblib.dump({"imp":imp,"sc":scl,"feats":mfeats,"imap":imap},outdir/"meta_log.jlb")

def main():
    for tag,dr in INPUT_VARIANTS:
        run(tag,dr,OUT_ROOT/tag)

if __name__=="__main__":
    main()