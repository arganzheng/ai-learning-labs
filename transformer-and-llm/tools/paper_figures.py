"""把预训练系列引用的原论文图裁出来（教学评述引用，图题注明来源与版权归原作者 / 出版方）。

    python tools/paper_figures.py                 # 全部
    python tools/paper_figures.py chinchilla-fig3

每个条目：arXiv PDF、页码（1 起）、裁剪框（PDF 点坐标 (x0, y0, x1, y1)）、dpi。
输出 out/paper-<name>.png，博客侧再转 WebP。PDF 缓存在 data/papers/。
"""
import os
import sys
import urllib.request

import pymupdf

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PDFS = os.path.join(HERE, "data", "papers")
OUT = os.path.join(HERE, "out")

FIGS = {
    # 预训练 03 scaling law
    "kaplan-fig1": ("https://arxiv.org/pdf/2001.08361", 3, (85, 65, 530, 205), 220),        # 三条幂律：算力 / 数据 / 参数
    "chinchilla-fig1": ("https://arxiv.org/pdf/2203.15556", 2, (55, 60, 540, 290), 220),    # 三种方法的预测叠在一起 + Kaplan 的线
    "chinchilla-fig2": ("https://arxiv.org/pdf/2203.15556", 5, (55, 60, 540, 208), 220),    # 方法 1：训练曲线包络
    "chinchilla-fig3": ("https://arxiv.org/pdf/2203.15556", 6, (55, 60, 540, 222), 220),    # 方法 2：IsoFLOP 曲线
    # 预训练 04 数据工程
    "fineweb-fig1-2": ("https://arxiv.org/pdf/2406.17557", 4, (100, 314, 512, 452), 220),   # 图 1 WARC+trafilatura vs WET；图 2 过滤 vs 不过滤（28B token 消融）
    "fineweb-fig3-4": ("https://arxiv.org/pdf/2406.17557", 5, (100, 200, 512, 342), 220),   # 图 3 全局去重反而更差；图 4 单快照上全局去重删掉的 vs 留下的
}


def fetch(src):
    os.makedirs(PDFS, exist_ok=True)
    name = src.rstrip("/").split("/")[-1].replace(".pdf", "") + ".pdf"
    path = os.path.join(PDFS, name)
    if not os.path.exists(path):
        print(f"  下载 {src} ...")
        urllib.request.urlretrieve(src, path)
    return path


def crop(name, src, page_no, clip, dpi):
    doc = pymupdf.open(fetch(src)); page = doc[page_no - 1]
    os.makedirs(OUT, exist_ok=True)
    out = os.path.join(OUT, f"paper-{name}.png")
    if clip is None:
        info = page.get_image_info(xrefs=True)[0]
        pix = pymupdf.Pixmap(doc, info["xref"])
        if pix.n - pix.alpha >= 4:
            pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
        pix.save(out)
        print(f"  {name}: 第 {page_no} 页嵌入图 {pix.width}×{pix.height} → out/paper-{name}.png")
    else:
        pix = page.get_pixmap(dpi=dpi, clip=pymupdf.Rect(*clip))
        pix.save(out)
        print(f"  {name}: 第 {page_no} 页裁剪 {clip} @ {dpi} dpi → {pix.width}×{pix.height} → out/paper-{name}.png")


if __name__ == "__main__":
    names = sys.argv[1:] or list(FIGS)
    for n in names:
        crop(n, *FIGS[n])
