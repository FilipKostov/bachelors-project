import os
import time
import json
import re
from pathlib import Path
import requests
from bs4 import BeautifulSoup
from urllib.parse import quote
import pandas as pd
db_conn="dbname=steam user=postgres password=password host=127.0.0.1 port=5432"
all_items_file="F:/steam_market_project/data/all_items.txt"
item_dim="F:/steam_market_project/data/item_dim.csv"
item_dim_summary="F:/steam_market_project/data/item_dim_summary_final.csv"
cache_file=Path("F:/steam_market_project/data/tags_cache.json")
def get_cookies():
    cookie=os.getenv("STEAM_LOGIN_SECURE","").strip()
    if not cookie:raise ValueError("STEAM_LOGIN_SECURE not set")
    return {"steamLoginSecure":cookie}
def request_with_backoff(url,cookies):
    attempt=0
    while attempt<5:
        try:
            r=requests.get(url,headers={"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},cookies=cookies,timeout=30)
            if r.status_code==429:time.sleep(30);attempt+=1;continue
            r.raise_for_status()
            return r
        except:time.sleep(10);attempt+=1
    return None
def load_all_items():
    with open(all_items_file,"r",encoding="utf-8") as f:return [line.strip() for line in f if line.strip()]
def load_cache():
    if cache_file.exists():return json.loads(cache_file.read_text(encoding="utf-8"))
    return {}
def save_cache(cache):
    cache_file.write_text(json.dumps(cache,ensure_ascii=False,indent=2),encoding="utf-8")
def extract_tags_from_page(text):
    tags=dict()
    text=text.lower()
    if "consumer grade" in text:
        tags["rarity"]="Consumer Grade"
    elif "industrial grade" in text:
        tags["rarity"]="Industrial Grade"
    elif "mil-spec grade" in text:
        tags["rarity"]="Mil-Spec Grade"
    elif "restricted" in text:
        tags["rarity"]="Restricted"
    elif "classified" in text:
        tags["rarity"]="Classified"
    elif "covert" in text:
        tags["rarity"]="Covert"
    m=re.search(r"exterior:\s*([\w\s-]+)",text,re.I)
    if m:tags["exterior"]=m.group(1).strip().title()
    m=re.search(r"the\s+[\w\s]+?\s+collection",text,re.I)
    if m:tags["collection"]=m.group(0).strip().title()
    return tags
def fetch_tags(name,cache):
    if name in cache:return cache[name]
    url=f"https://steamcommunity.com/market/listings/730/{quote(name)}"
    try:
        cookies=get_cookies()
        r=request_with_backoff(url,cookies)
        if not r:return {"item_type":"NA","weapon_class":"NA","rarity":"NA","collection":"NA","quality":"NA","wear_full":"NA","sticker_type":"NA"}
        soup=BeautifulSoup(r.text,"html.parser")
        text=r.text
    except:
        return {"item_type":"NA","weapon_class":"NA","rarity":"NA","collection":"NA","quality":"NA","wear_full":"NA","sticker_type":"NA"}
    raw=extract_tags_from_page(text)
    info={"item_type":"NA","weapon_class":"NA","rarity":raw.get("rarity","NA"),"collection":raw.get("collection","NA"),"quality":"NA","wear_full":raw.get("exterior","NA"),"sticker_type":"NA"}
    cache[name]=info
    return info
def main():
    items=load_all_items()
    cache=load_cache()
    out=list()
    for idx in range(len(items)):
        name=items[idx]
        t=fetch_tags(name,cache)
        base_item=name
        if t["weapon_class"]!="NA":
            base_family=f"{t['weapon_class']} | {t['item_type']}"
        else:
            base_family=name
        if name.startswith("StatTrak™ "):
            is_stattrak=1
        else:
            is_stattrak=0
        if name.startswith("Souvenir "):
            is_souvenir=1
        else:
            is_souvenir=0
        r={"item_name":name,"base_item":base_item,"base_family":base_family,"item_type":t["item_type"],"weapon_class":t["weapon_class"],"rarity":t["rarity"],"collection":t["collection"],"quality":t["quality"],"wear_full":t["wear_full"],"sticker_type":t["sticker_type"],"is_stattrak":is_stattrak,"is_souvenir":is_souvenir}
        out.append(r)
        time.sleep(6.0)
    dim=pd.DataFrame(out)
    Path(item_dim).parent.mkdir(parents=True,exist_ok=True)
    dim.to_csv(item_dim,index=False,encoding="utf-8")
    save_cache(cache)
    summary=pd.DataFrame([(c,int((dim[c]!="NA").sum()),int((dim[c]=="NA").sum()),float((dim[c]!="NA").mean()*100)) for c in dim.columns],columns=["metric","present","missing","percent_present"])
    summary.to_csv(item_dim_summary,index=False,encoding="utf-8")
    print(summary)
if __name__=="__main__":
    main()