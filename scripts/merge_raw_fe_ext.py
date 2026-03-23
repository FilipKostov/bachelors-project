from pathlib import Path
import pandas as pd

fe_dir=Path("F:/steam_market_project/data/model_subset_csv_fe_no_cat")
ext_dir=Path("F:/steam_market_project/data/external")
out_dir=Path("F:/steam_market_project/data/model_subset_csv_raw_fe_liquipedia")
out_dir.mkdir(parents=True, exist_ok=True)

ext_files=sorted(list(ext_dir.glob("idx_liquipedia_*.csv")) +list(ext_dir.glob("liquipedia_daily_*.csv")))

ext_dfs=list()
for f in ext_files:
    try:
        df_tmp=pd.read_csv(f)
        ext_dfs.append(df_tmp)
    except Exception as e:
        print(f"Failed")

if ext_dfs:
    ext_df=pd.concat(ext_dfs, ignore_index=True)
    ext_df["date"]=ext_df["date"].astype(str)
else:
    ext_df=pd.DataFrame(columns=["date"])

def merge_and_save_one_file(input_path: Path, output_dir: Path):
    df=pd.read_csv(input_path)
    df["date"]=df["date"].astype(str)
    merged=df.merge(ext_df, on="date", how="left")
    merged=merged.sort_values(["item_name", "date"]).reset_index(drop=True)
    out_path=output_dir / input_path.name
    merged.to_csv(out_path, index=False)

fe_files=sorted(fe_dir.glob("subset_*.csv"))

for i, f in enumerate(fe_files, 1):
    merge_and_save_one_file(f, out_dir)
