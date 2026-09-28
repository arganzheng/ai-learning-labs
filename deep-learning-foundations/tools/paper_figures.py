"""把原论文里的经典结构图裁出来（教学评述引用，图题注明来源与版权归原作者 / 出版方）。

    python tools/paper_figures.py            # 全部
    python tools/paper_figures.py xiong-preln adam-alg1

每个条目：arXiv id（或 URL）、页码（1 起）、裁剪框（PDF 点坐标，(x0, y0, x1, y1)；None = 页面里第一张嵌入图）。
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
    # name: (source, page, clip or None, dpi)
    "xiong-preln": ("https://arxiv.org/pdf/2002.04745", 2, None, 220),                     # Fig.1 Post-LN vs Pre-LN（嵌入位图）
    "adam-alg1": ("https://arxiv.org/pdf/1412.6980", 2, (105, 105, 507, 362), 220),        # Algorithm 1
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
