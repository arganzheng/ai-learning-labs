"""案例用的真实数据集：第一次运行下载到 data/（已 gitignore），之后离线。

    from _data import california_housing, sms_spam
    df = california_housing()        # 20,640 行 × 10 列（Géron《Hands-On ML》同款，含 ocean_proximity 与缺失值）
    df = sms_spam()                   # 5,574 行：label ∈ {ham, spam}, text

sklearn 自带的 fetch_california_housing / fetch_olivetti_faces 走 figshare，有些网络下 403，
这里一律走 openml（同一份数据，多了几列）或 UCI 的静态包。
"""
import io
import os
import urllib.request
import zipfile

import pandas as pd

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def _openml(data_id, name):
    """openml 数据集 → DataFrame，缓存为 data/<name>.parquet。"""
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, f"{name}.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)
    from sklearn.datasets import fetch_openml
    print(f"  下载 openml #{data_id} ({name}) ...")
    df = fetch_openml(data_id=data_id, as_frame=True, data_home=DATA).frame
    df.to_parquet(path)
    return df


def _uci_zip(url, member, name):
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, name)
    if not os.path.exists(path):
        print(f"  下载 {url} ...")
        with urllib.request.urlopen(url, timeout=120) as r:
            z = zipfile.ZipFile(io.BytesIO(r.read()))
        with open(path, "wb") as f:
            f.write(z.read(member))
    return path


def california_housing():
    """加州 1990 年人口普查，每行一个街区（block group）。目标 median_house_value（美元，50 万处截断）。
    openml #43939 = Kaggle "California Housing Prices" = Géron《Hands-On ML》第 2 章的数据；
    sklearn 自带版本少了 ocean_proximity 并已把 total_* 折成人均量。CC0。"""
    df = _openml(43939, "california_housing")
    df["ocean_proximity"] = df["ocean_proximity"].astype(str)
    return df


def sms_spam():
    """UCI SMS Spam Collection（Almeida & Hidalgo 2011）：5,574 条英文短信，747 条 spam（13.4%）。CC BY 4.0。"""
    path = _uci_zip("https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip",
                    "SMSSpamCollection", "SMSSpamCollection.tsv")
    return pd.read_csv(path, sep="\t", header=None, names=["label", "text"], quoting=3)
