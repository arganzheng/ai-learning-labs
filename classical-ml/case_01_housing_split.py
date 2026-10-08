"""案例（经典 ML 01）：加州房价——同一个模型，随机划分 vs 按地区划分，差多少。
https://arganzheng.life/what-is-learning-splits-generalization-and-bias-variance.html

    python case_01_housing_split.py

问题：划分方式必须模仿"模型上线后会遇到什么数据"。如果模型要给**没见过的地区**估价，
随机划分会把同一条街的邻居留在训练集里，KNN 靠"抄邻居"拿高分，上线后失效。
图输出到 out/case-01-*.svg。
"""
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    cross_val_score,
    train_test_split,
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from _data import california_housing
from _plot import C, plt, save

NUM = ["longitude", "latitude", "housing_median_age", "total_rooms", "total_bedrooms",
       "population", "households", "median_income"]


def rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def models():
    return {
        "KNN(k=10)，只用经纬度": (["longitude", "latitude"],
                                make_pipeline(StandardScaler(), KNeighborsRegressor(10))),
        "KNN(k=10)，全部 8 个特征": (NUM, make_pipeline(SimpleImputer(), StandardScaler(), KNeighborsRegressor(10))),
        "线性回归，全部 8 个特征": (NUM, make_pipeline(SimpleImputer(), StandardScaler(), LinearRegression())),
    }


def main():
    df = california_housing()
    y = df["median_house_value"].values
    print(f"数据：{len(df)} 个街区，目标 = 街区房价中位数，均值 {y.mean():,.0f}，标准差 {y.std():,.0f} 美元")

    # 两种划分：随机 20% vs 按 1°×1° 网格整块留出（模拟"新地区"）
    idx_all = np.arange(len(df))
    tr_r, te_r = train_test_split(idx_all, test_size=0.2, random_state=0)
    cell = (np.floor(df["longitude"]).astype(int).astype(str) + "," + np.floor(df["latitude"]).astype(int).astype(str))
    tr_g, te_g = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=3).split(idx_all, groups=cell))
    print(f"随机划分：训练 {len(tr_r)} / 测试 {len(te_r)}；按地区划分：{cell.nunique()} 个 1°×1° 格子，"
          f"留出 {cell.iloc[te_g].nunique()} 个格子共 {len(te_g)} 个街区做测试")

    print(f"\n{'模型':<28}{'随机划分 RMSE':>14}{'按地区划分 RMSE':>16}{'变差':>8}")
    base = rmse(y[te_r], np.full(len(te_r), y[tr_r].mean()))
    print(f"{'基线：永远猜训练集均值':<28}{base:>14,.0f}{rmse(y[te_g], np.full(len(te_g), y[tr_g].mean())):>16,.0f}")
    results = {}
    for name, (cols, m) in models().items():
        X = df[cols].values
        r = rmse(y[te_r], m.fit(X[tr_r], y[tr_r]).predict(X[te_r]))
        g = rmse(y[te_g], m.fit(X[tr_g], y[tr_g]).predict(X[te_g]))
        results[name] = (r, g)
        print(f"{name:<28}{r:>14,.0f}{g:>16,.0f}{g / r:>7.2f}×")
    print("\n解读：随机划分下 KNN 只用经纬度就是三者最好——它在抄同一条街邻居的价格；"
          "换成整块地区留出，邻居不在训练集里，它一下子变成最差。"
          "用全部 8 个特征的 KNN 与线性回归几乎不受影响：它们学的是'收入高的街区房价高'这类能带到新地区的规律。"
          "哪个数字是'真的'，取决于模型上线后要给哪种数据估价。")

    # 图 1：两种划分在地图上长什么样
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.4), sharex=True, sharey=True)
    for ax, (tr, te, title) in zip(axes, [(tr_r, te_r, "随机划分：测试点混在训练点里"),
                                           (tr_g, te_g, "按地区划分：整块地区留出")]):
        ax.scatter(df.longitude.iloc[tr], df.latitude.iloc[tr], s=1, c=C["light"], label="训练")
        ax.scatter(df.longitude.iloc[te], df.latitude.iloc[te], s=1, c=C["red"], label="测试")
        ax.set_title(title); ax.set_xlabel("经度"); ax.set_aspect("equal")
        ax.legend(markerscale=6, loc="upper right")
    axes[0].set_ylabel("纬度")
    save(fig, "case-01-split-map")

    # 图 2：RMSE 对比
    fig, ax = plt.subplots(figsize=(7.6, 2.8))
    names = list(results)
    x = np.arange(len(names)); w = 0.36
    ax.bar(x - w / 2, [results[n][0] / 1e3 for n in names], w, color=C["blue"], label="随机划分")
    ax.bar(x + w / 2, [results[n][1] / 1e3 for n in names], w, color=C["red"], label="按地区划分")
    ax.axhline(base / 1e3, ls="--", c=C["gray"], lw=1); ax.text(2.45, base / 1e3 + 2, "基线（猜均值）", ha="right", fontsize=8, color=C["gray"])
    ax.set_xticks(x); ax.set_xticklabels(names); ax.set_ylabel("测试 RMSE（千美元）"); ax.legend()
    save(fig, "case-01-split-rmse")

    # 追加：训练集内按组交叉验证选 k —— 预处理与模型都只在训练折上 fit，测试集只在最后看一次
    print("\n训练集内按组 5 折交叉验证选 KNN 的 k（只用按地区划分的训练集，每一折按 1°×1° 格子切）：")
    X8 = df[NUM].values
    imp = SimpleImputer().fit(X8[tr_g])                                                     # !ref imp_fit
    print(f"total_bedrooms 的填补值：训练集均值 {imp.statistics_[4]:,.1f}，全部数据均值 {np.nanmean(X8[:, 4]):,.1f}"
          " —— Pipeline 里 SimpleImputer / StandardScaler 的统计量只来自 fit 时看到的那份数据")
    print(f"{'k':>5}{'按组 5 折 CV RMSE':>18}{'事后对照：留出地区测试 RMSE':>26}")
    cv_rmse = {}
    for k in (3, 10, 30, 100, 300):
        m = make_pipeline(SimpleImputer(), StandardScaler(), KNeighborsRegressor(k))
        scores = cross_val_score(m, X8[tr_g], y[tr_g], groups=cell.iloc[tr_g], cv=GroupKFold(5),
                                 scoring="neg_root_mean_squared_error")                       # !ref cv_group
        cv_rmse[k] = -scores.mean()
        t = rmse(y[te_g], m.fit(X8[tr_g], y[tr_g]).predict(X8[te_g]))
        print(f"{k:>5}{cv_rmse[k]:>18,.0f}{t:>26,.0f}")
    k_best = min(cv_rmse, key=lambda k: cv_rmse[k])
    final = make_pipeline(SimpleImputer(), StandardScaler(), KNeighborsRegressor(k_best)).fit(X8[tr_g], y[tr_g])
    print(f"CV 选出 k={k_best}；用整个训练集重新 fit 后，留出地区测试 RMSE {rmse(y[te_g], final.predict(X8[te_g])):,.0f}"
          "（右列只是事后对照，选 k 时没有看它）")


if __name__ == "__main__":
    main()
