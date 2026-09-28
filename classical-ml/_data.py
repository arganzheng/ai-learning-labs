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


def titanic():
    """泰坦尼克 1,309 名乘客（openml #40945，Vanderbilt 版全量数据）。目标 survived。
    注意 boat（救生艇号）与 body（遗体编号）是事后才知道的——放进特征就是泄漏。公有领域。"""
    df = _openml(40945, "titanic")
    df["survived"] = df["survived"].astype(int)
    for c in ["sex", "embarked"]:
        df[c] = df[c].astype(str)
    return df


def adult():
    """UCI Adult / Census Income（openml #1590）：1994 年美国人口普查 48,842 人，14 个特征，目标：年收入是否 >50K（23.9%）。CC BY 4.0。"""
    df = _openml(1590, "adult")
    df["class"] = (df["class"].astype(str) == ">50K").astype(int)
    return df


def mnist(kind="train"):
    """MNIST 手写数字：X [n, 784] float32 ∈ [0, 1]，y [n]。复用 deep-learning-foundations/dlf/data.py 的下载与缓存。"""
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(DATA), "..", "deep-learning-foundations"))
    from dlf.data import load_mnist
    return load_mnist(kind)


BANK_COLS = ["age", "job", "marital", "education", "default", "balance", "housing", "loan", "contact", "day",
             "month", "duration", "campaign", "pdays", "previous", "poutcome", "y"]


def bank_marketing():
    """UCI Bank Marketing（openml #1461，Moro 等 2014）：葡萄牙银行 45,211 次电话营销，目标：客户是否订了定期存款（11.7%）。
    duration（通话时长）只有打完电话才知道——用它预测就是泄漏，脚本里默认去掉。CC BY 4.0。"""
    df = _openml(1461, "bank_marketing")
    df.columns = BANK_COLS
    df["y"] = (df["y"].astype(str) == "2").astype(int)
    for c in df.columns:
        if str(df[c].dtype) == "category":
            df[c] = df[c].astype(str)
    return df


def olivetti_faces():
    """Olivetti / AT&T 人脸（openml #41083）：40 个人 × 10 张 = 400 张 64×64 灰度图。返回 X [400, 4096] ∈ [0, 1]，y [400]。"""
    import numpy as np
    df = _openml(41083, "olivetti_faces")
    X = df.iloc[:, :-1].to_numpy(dtype="float32")
    y = df.iloc[:, -1].astype(int).to_numpy()
    return X, y


def online_retail():
    """UCI Online Retail（Chen 2012）：一家英国网店 2010-12 到 2011-12 的 541,909 行交易记录（发票号、商品、数量、单价、客户号、国家）。
    首次运行下载 23 MB 的 xlsx，转成 parquet 缓存。CC BY 4.0。"""
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, "online_retail.parquet")
    if not os.path.exists(path):
        xlsx = _uci_zip("https://archive.ics.uci.edu/static/public/352/online+retail.zip", "Online Retail.xlsx", "Online Retail.xlsx")
        print("  读取 xlsx（约一分钟）...")
        df = pd.read_excel(xlsx)
        df.columns = ["invoice", "stock_code", "description", "quantity", "invoice_date", "unit_price", "customer_id", "country"]
        for c in ["invoice", "stock_code", "description", "country"]:
            df[c] = df[c].astype(str)
        df.to_parquet(path)
    return pd.read_parquet(path)


def wikitext2(split="train"):
    """wikitext-2-raw 的段落列表（本地 HF 缓存 Salesforce/wikitext；没有缓存时用 datasets 下载约 4 MB）。"""
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
    return [t.strip() for t in ds["text"] if len(t.strip()) > 200 and not t.strip().startswith("=")]
