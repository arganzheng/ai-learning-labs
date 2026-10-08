"""案例（经典 ML 02）：加州房价预测——从"猜均值"到 Ridge / Lasso，一步一步看误差怎么降。
https://arganzheng.life/linear-regression-least-squares-ridge-and-lasso.html

    python case_02_housing_regression.py

数据：20,640 个街区，8 个数值特征 + 1 个类别特征（离海距离），目标是街区房价中位数。
图输出到 out/case-02-*.svg。
"""
import time

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler

from _data import california_housing
from _plot import C, plt, save

NUM = ["longitude", "latitude", "housing_median_age", "total_rooms", "total_bedrooms",
       "population", "households", "median_income"]


def rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


LONG_TAIL = ["total_rooms", "total_bedrooms", "population", "households",
             "rooms_per_household", "bedrooms_per_room", "population_per_household"]


def add_features(df, log=False):
    """三个"人均量"：街区的 total_rooms 取决于街区有多大，除以户数才是房子大不大。
    log=True 时把长尾列取 log1p：一个 39,320 间房的街区不该把直线拽歪。"""
    df = df.copy()
    df["rooms_per_household"] = df.total_rooms / df.households
    df["bedrooms_per_room"] = df.total_bedrooms / df.total_rooms
    df["population_per_household"] = df.population / df.households
    if log:
        for c in LONG_TAIL:
            df[c] = np.log1p(df[c])
    return df


def preprocess(num_cols, cat=True, poly=1):
    """数值：补缺失 → （可选）二次多项式 → 标准化；类别：one-hot。"""
    num = make_pipeline(SimpleImputer(strategy="median"),
                        *([PolynomialFeatures(poly, include_bias=False)] if poly > 1 else []),
                        StandardScaler())
    parts = [("num", num, num_cols)]
    if cat:
        parts.append(("cat", OneHotEncoder(handle_unknown="ignore"), ["ocean_proximity"]))
    return ColumnTransformer(parts)


def main():
    raw = add_features(california_housing())
    y = raw.pop("median_house_value").values
    logd = add_features(california_housing(), log=True).drop(columns="median_house_value")
    tr, te = train_test_split(np.arange(len(raw)), test_size=0.2, random_state=0)
    ytr, yte = y[tr], y[te]
    print(f"数据：{len(raw)} 个街区；训练 {len(tr)}，测试 {len(te)}；目标均值 {y.mean():,.0f}，标准差 {y.std():,.0f} 美元")
    print(f"缺失：total_bedrooms 缺 {raw.total_bedrooms.isna().sum()} 个（用中位数补）；"
          f"ocean_proximity 5 类：{raw.ocean_proximity.value_counts().to_dict()}")
    print(f"长尾：total_rooms 中位数 {raw.total_rooms.median():,.0f}，最大 {raw.total_rooms.max():,.0f}；"
          f"rooms_per_household 中位数 {raw.rooms_per_household.median():.1f}，最大 {raw.rooms_per_household.max():.0f}")

    NUM2 = NUM + ["rooms_per_household", "bedrooms_per_room", "population_per_household"]
    ALL = NUM2 + ["ocean_proximity"]
    alphas = np.logspace(-2, 3, 20)
    steps = [
        ("0 基线：永远猜训练集均值", None, None, raw),
        ("1 线性回归：8 个原始数值特征", make_pipeline(preprocess(NUM, cat=False), LinearRegression()), NUM, raw),
        ("2 + 离海距离 one-hot（5 列）", make_pipeline(preprocess(NUM), LinearRegression()), NUM + ["ocean_proximity"], raw),
        ("3 + 三个人均量特征", make_pipeline(preprocess(NUM2), LinearRegression()), ALL, raw),
        ("4 + 长尾列取 log", make_pipeline(preprocess(NUM2), LinearRegression()), ALL, logd),
        ("5 二次多项式（77 列）", make_pipeline(preprocess(NUM2, poly=2), LinearRegression()), ALL, logd),
        ("6 三次多项式（363 列），无正则", make_pipeline(preprocess(NUM2, poly=3), LinearRegression()), ALL, logd),
        ("7 三次多项式 + Ridge（α 交叉验证）", make_pipeline(preprocess(NUM2, poly=3), RidgeCV(alphas=alphas)), ALL, logd),
        ("8 三次多项式 + Lasso(α=100)", make_pipeline(preprocess(NUM2, poly=3), Lasso(alpha=100, max_iter=50000)), ALL, logd),
        ("9 对照：梯度提升树", make_pipeline(preprocess(NUM2), HistGradientBoostingRegressor(max_iter=500, learning_rate=0.1, random_state=0)), ALL, logd),
    ]
    print(f"\n{'步骤':<40}{'训练 RMSE':>10}{'测试 RMSE':>10}{'R²':>7}{'耗时':>7}")
    res = {}
    for name, model, cols, data in steps:
        t = time.time()
        if model is None:
            ptr, pte = np.full(len(tr), ytr.mean()), np.full(len(te), ytr.mean())
        else:
            m = model.fit(data.iloc[tr][cols], ytr)                    # 只用 cols 里的列，其余列对这一步不可见
            ptr, pte = m.predict(data.iloc[tr][cols]), m.predict(data.iloc[te][cols])
            res[name] = (m, pte)
        r2 = 1 - np.mean((yte - pte) ** 2) / np.var(yte)
        extra = f"  α={m[-1].alpha_:.2f}" if model is not None and hasattr(m[-1], "alpha_") else ""
        print(f"{name:<40}{rmse(ytr, ptr):>10,.0f}{rmse(yte, pte):>10,.0f}{r2:>7.3f}{time.time() - t:>6.1f}s{extra}")

    # 第 3 步的系数：标准化后的每个特征 ±1 个标准差，房价变多少
    m3 = res["3 + 三个人均量特征"][0]
    names = list(m3[0].get_feature_names_out())
    coef = m3[-1].coef_
    print("\n第 3 步的系数（特征已标准化，单位：美元 / 标准差）：")
    for n, c in sorted(zip(names, coef), key=lambda t: -abs(t[1])):
        print(f"  {n:<36}{c:>+10,.0f}")
    print("解读：收入每高一个标准差（约 1.9 万美元），房价高约 7 万；经度、纬度系数都是负的大数——"
          "它们合起来说的是'越往西北越贵'（旧金山、湾区），这是一条直线能表达'位置'的极限。")
    m4 = res["4 + 长尾列取 log"][0]
    c4 = dict(zip(m4[0].get_feature_names_out(), m4[-1].coef_))
    print("\n第 4 步取 log 之后，log(population) − log(households) = log(population_per_household)，三列严格共线，系数变成：")
    for k in ["num__population", "num__households", "num__population_per_household"]:
        print(f"  {k:<36}{c4[k]:>+10,.0f}")
    print("三个巨大的数互相抵消，预测没变差（测试 RMSE 还降了），但系数不能再当'每个特征的价格'读——这是第五章说的共线。")

    # Lasso 砍掉了多少列
    c6 = res["8 三次多项式 + Lasso(α=100)"][0][-1].coef_
    c7 = res["7 三次多项式 + Ridge（α 交叉验证）"][0][-1].coef_
    print(f"\nLasso：{len(c6)} 列里 {np.sum(c6 == 0)} 列系数恰好为 0，只留下 {np.sum(c6 != 0)} 列；"
          f"Ridge 的 {len(c7)} 列有 {np.sum(c7 == 0)} 列为 0——它只把系数压小，不清零。")

    # 图 1：测试 RMSE 逐步下降
    fig, ax = plt.subplots(figsize=(7.6, 3.0))
    vals = []
    for name, model, cols, data in steps:
        pte = res[name][1] if name in res else np.full(len(te), ytr.mean())
        vals.append(rmse(yte, pte) / 1e3)
    colors = [C["gray"]] + [C["blue"]] * 4 + [C["orange"]] + [C["red"]] + [C["green"]] * 2 + [C["purple"]]
    ax.bar(range(len(vals)), vals, color=colors)
    for i, v in enumerate(vals):
        ax.text(i, min(v, 118) + 1.5, f"{v:.0f}", ha="center", fontsize=8)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(["猜均值", "8 特征\n线性", "+离海\n距离", "+人均量", "+长尾\n取 log", "二次项\n77 列", "三次项\n363 列\n无正则", "三次项\nRidge", "三次项\nLasso", "梯度\n提升树"])
    ax.set_ylabel("测试 RMSE（千美元）"); ax.set_ylim(0, 125)
    ax.set_title("每加一步，测试误差降多少")
    save(fig, "case-02-rmse-steps")

    # 图 2：预测 vs 真实（第 5 步 Ridge）
    pte = res["7 三次多项式 + Ridge（α 交叉验证）"][1]
    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    ax.scatter(yte / 1e3, pte / 1e3, s=2, alpha=0.3, c=C["blue"])
    ax.plot([0, 500], [0, 500], c=C["red"], lw=1)
    ax.set_xlabel("真实房价（千美元）"); ax.set_ylabel("预测房价（千美元）")
    ax.set_title(f"三次多项式 + Ridge：测试 RMSE {rmse(yte, pte) / 1e3:.0f} 千美元")
    ax.annotate("50 万处的竖线：\n数据把房价截断在 500,001", xy=(500, 150), xytext=(320, 20), fontsize=8,
                color=C["gray"], arrowprops={"arrowstyle": "->", "color": C["gray"]})
    save(fig, "case-02-pred-vs-true")

    # 图 3：第 3 步系数
    fig, ax = plt.subplots(figsize=(7.6, 3.2))
    order = np.argsort(coef)
    ax.barh([names[i].replace("num__", "").replace("cat__ocean_proximity_", "海: ") for i in order],
            coef[order] / 1e3, color=[C["red"] if c < 0 else C["blue"] for c in coef[order]])
    ax.set_xlabel("系数（千美元 / 标准差）"); ax.set_title("线性回归学到的每个特征的'价格'")
    save(fig, "case-02-coefficients")

    # 追加：第 6 步为什么不稳——363 列设计矩阵的条件数，inv / solve / lstsq 三种解法差多少；α 在求和式与平均式之间的换算
    m6 = res["6 三次多项式（363 列），无正则"][0]
    Xtr6 = m6[0].transform(logd.iloc[tr][ALL])
    Xte6 = m6[0].transform(logd.iloc[te][ALL])
    Xtr6 = np.asarray(Xtr6.todense()) if hasattr(Xtr6, "todense") else np.asarray(Xtr6)
    Xte6 = np.asarray(Xte6.todense()) if hasattr(Xte6, "todense") else np.asarray(Xte6)
    Xb_tr, Xb_te = np.c_[np.ones(len(Xtr6)), Xtr6], np.c_[np.ones(len(Xte6)), Xte6]
    G = Xb_tr.T @ Xb_tr
    print(f"\n第 6 步的设计矩阵（含截距列）：{Xb_tr.shape}，cond(X) = {np.linalg.cond(Xb_tr):.1e}，cond(XᵀX) = {np.linalg.cond(G):.1e}；"
          f"5 列 one-hot 之和恒等于截距列，XᵀX 在精确算术里奇异；sklearn 报告的秩 rank_ = {m6[-1].rank_}（共 {Xtr6.shape[1]} 列，不含截距）")
    sols = []
    with np.errstate(all="ignore"):
        for name, fn in (("inv(XᵀX)·Xᵀy", lambda: np.linalg.inv(G) @ (Xb_tr.T @ ytr)),
                         ("solve(XᵀX, Xᵀy)", lambda: np.linalg.solve(G, Xb_tr.T @ ytr)),
                         ("lstsq(X, y)", lambda: np.linalg.lstsq(Xb_tr, ytr, rcond=None)[0])):
            try:
                w = fn()
                sols.append((name, rmse(ytr, Xb_tr @ w), rmse(yte, Xb_te @ w), float(np.abs(w).max())))
            except np.linalg.LinAlgError as e:
                sols.append((name, float("nan"), float("nan"), float("nan")))
                print(f"  {name:<18} LinAlgError: {e}")
    print(f"  {'解法':<18}{'训练 RMSE':>16}{'测试 RMSE':>18}{'max|w|':>10}")
    for name, a, b, c in sols:
        print(f"  {name:<18}{a:>16,.0f}{b:>18,.0f}{c:>10.1e}")
    pte6 = res["6 三次多项式（363 列），无正则"][1]
    print(f"  sklearn LinearRegression（scipy lstsq）测试 RMSE {rmse(yte, pte6):,.0f}；"
          "无正则 + 近奇异，哪个解法、哪个 BLAS 都可能给出不同的测试误差，这一行的数字换台机器就会变")
    a_ridge = res["7 三次多项式 + Ridge（α 交叉验证）"][0][-1].alpha_
    n_tr = len(tr)
    print(f"  α 的换算：RidgeCV 选的 α={a_ridge:.2f} 作用在求和式 ‖y−Xw‖² + α‖w‖² 上，换成平均式 (1/n)‖y−Xw‖² + λ‖w‖² 是 λ = α/n = {a_ridge / n_tr:.2e}（n={n_tr}）；"
          f"Lasso(α=100) 的目标是 (1/2n)‖y−Xw‖² + α‖w‖₁，换成求和式 ‖y−Xw‖² + β‖w‖₁ 是 β = 2nα = {2 * n_tr * 100:,.0f}")


if __name__ == "__main__":
    main()
