"""Shared command line: `python 0X_xxx.py [--quick] [exp ...]`."""
import argparse, sys, time

def parse(experiments, description=""):
    """experiments: dict name -> one-line description. Returns (selected names, quick flag)."""
    ap = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog="experiments:\n" + "\n".join(f"  {k:6s} {v}" for k, v in experiments.items()))
    ap.add_argument("exp", nargs="*", default=list(experiments), help="which experiments to run (default: all)")
    ap.add_argument("--quick", action="store_true", help="shrink steps/epochs so the whole script finishes in ~1 min (numbers will differ from the post)")
    a = ap.parse_args()
    bad = [e for e in a.exp if e not in experiments]
    if bad: ap.error(f"unknown experiment(s) {bad}; choose from {list(experiments)}")
    if a.quick: print("[quick mode: reduced steps, numbers differ from the post]\n")
    return a.exp, a.quick

class Timer:
    def __init__(self): self.t0 = time.time()
    def __str__(self): return f"{time.time() - self.t0:.0f}s"
    def secs(self): return time.time() - self.t0
