"""12《系列总结与通关自测》toy：按各篇的公式从第一性原理重算必记表数字。

  python 12_recap_toys.py ledger   # 每项：重算值 vs 文中值，标 OK / MISMATCH
"""
import math

N_70 = 70.55e9      # 70B（字节账用，与第一、八篇一致）
N_70F = 69.5e9      # 70B（FLOP 账用，与第一篇的 449 GFLOP 一致）
N_405 = 405.85e9
S, B, H, L = 8192, 1, 8192, 80

checks = []


def check(name, got, stated, fmt="{:.3f}", tol=0.02):
    ok = abs(got - stated) / max(abs(stated), 1e-9) < tol
    checks.append((name, fmt.format(got), stated, "OK" if ok else "MISMATCH"))


# ---- 第一篇：两本账 ----
check("常驻 16N，405B（TB）", 16 * N_405 / 1e12, 6.49)
check("常驻 18N，405B（TB）", 18 * N_405 / 1e12, 7.3)
check("常驻 16N，70B（TB）", 16 * N_70 / 1e12, 1.13)
check("落盘 14N，405B（TB）", 14 * N_405 / 1e12, 5.7)
check("落盘 12N，70B（GB）", 12 * N_70 / 1e9, 847)
check("激活 34sbh，70B 每层（GB）", 34 * S * B * H / 1e9, 2.28)
flop_tok = 6 * N_70F + 6 * L * S * H
check("每 token 6N+6Lsh，70B（GFLOP）", flop_tok / 1e9, 449)
# 吞吐账用 6N+12Lsh（第七篇的 4.81e11），理论下限对 2.0 是取整
flop_tp = 6 * N_70F + 12 * L * S * H
check("每 token 6N+12Lsh，70B（GFLOP）", flop_tp / 1e9, 481)
t_min = 4e6 * flop_tp / (1024 * 989e12)
check("理论 step 下限（s，文中取整为 2.0）", t_min, 1.9, tol=0.02)

# ---- 第二/七篇：气泡 ----
check("候选 A 气泡 (p-1)/(vm)，p=4 v=4 m=16（%）",
      (4 - 1) / (4 * 16) * 100, 4.7)
check("候选 B 气泡 (p-1)/(vm)，p=2 v=4 m=8（%）",
      (2 - 1) / (4 * 8) * 100, 3.1)
check("405B 非交错气泡 (p-1)/m，p=16 m=16（%）",
      (16 - 1) / 16 * 100, 94)

# ---- 第八篇：Young 公式 ----
M = 3.1 * 3600
for delta, t_stated, w_stated in ((600, 61, 33), (60, 19, 10.4),
                                  (10, 8, 4.2), (2, 3.5, 1.9)):
    t = math.sqrt(2 * delta * M) / 60
    w = math.sqrt(2 * delta / M) * 100
    check(f"τ_opt，M=3.1h δ={delta}s（min）", t, t_stated)
    check(f"最小浪费，δ={delta}s（%）", w, w_stated)

# ---- 第九篇：五项公式，16K 卡三场景 ----
M16 = 10980  # s


def G(Td, Tr, Tl, tau, delta):
    return (1 - (Td + Tr + Tl + tau / 2) / M16) / (1 + delta / tau) * 100


check("G 场景 A 同步（%）", G(600, 300, 120, 2580, 300), 70.8)
check("G 场景 B 异步（%）", G(600, 300, 120, 468, 10), 86.7)
check("G 场景 C 异步+快检测重启（%）", G(60, 30, 60, 468, 10), 94.5)

# ---- 第十一篇：成本 ----
check("1 个 MFU 点 = 30 天 / 42（天）", 30 / 42, 0.71)
check("1 个 MFU 点 = 0.714 天 × 1024 卡 × $2.5/h（美元）",
      30 / 42 * 1024 * 2.5 * 24, 43750)
check("1024 卡一天（万美元）", 1024 * 2.5 * 24 / 1e4, 6.1)
check("告警+runbook 每次省 30 min（GPU 小时）", 0.5 * 1024, 512)


def cmd_ledger():
    print("必记数字的重算（公式来自对应各篇，值来自总结表）：\n")
    for name, got, stated, tag in checks:
        print(f"  [{tag:>8}] {name}：重算 {got} vs 文中 {stated}")
    bad = [c for c in checks if c[3] != "OK"]
    print(f"\n{len(checks) - len(bad)}/{len(checks)} 项一致",
          f"{'；MISMATCH：' + '，'.join(c[0] for c in bad) if bad else ''}")


if __name__ == "__main__":
    cmd_ledger()
