"""预训练（04）「先讲明白」：一篇真实网页清洗前后的对照——WET 抽出来的正文（还带着导航、页脚、按钮）
与 C4 行级规则清理之后剩下的正文。挑一篇"正文像样、但被删掉了一半以上行"的英文网页打印出来。

    python show_before_after.py [第几篇候选，默认 1；3 号是一个「规则漏网」的例子]
依赖：step1 已跑过（data/raw.jsonl）。
"""
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from quality_filters import c4_lines, gopher_quality  # noqa: E402

from common import DATA, english_score, read_jsonl  # noqa: E402

which = int(sys.argv[1]) if len(sys.argv) > 1 else 1

if __name__ == "__main__":
    cands = []
    for d in read_jsonl(f"{DATA}/raw.jsonl"):
        t = d["text"]
        if not (600 < len(t) < 2500) or english_score(t) < 0.12:
            continue
        if any(not ok for ok, _ in gopher_quality(t).values()):
            continue
        keep, drop = c4_lines(t)
        if len(drop) >= 8 and len(keep) >= 3 and len(re.findall(r"\S+", "\n".join(keep))) >= 80:
            cands.append((d, keep, drop))
            if len(cands) > which:
                break
    d, keep, drop = cands[which]
    print(f"URL：{d['url']}\n")
    print(f"=== 清洗前：WET 抽出的正文，{len(d['text'].splitlines())} 行、{len(d['text'])} 字符 ===")
    for l in d["text"].splitlines():
        mark = "  " if l in keep else "✗ "
        print(f"{mark}{l[:100]}")
    print(f"\n=== 清洗后：C4 行级规则留下 {len(keep)} 行、{len(chr(10).join(keep))} 字符（删掉 {len(drop)} 行）===")
    for l in keep:
        print(f"  {l[:100]}")
    print("\n✗ 的行被删：不以句末标点结尾、不足 3 个词、或含 javascript / cookie / lorem ipsum 之类的词——导航、按钮、页脚、版权行。")
