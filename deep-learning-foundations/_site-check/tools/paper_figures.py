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
    # DL 04 正则化
    "dropout-fig1": ("https://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf", 2, (130, 85, 495, 258), 220),   # 标准网 vs 丢弃后的稀疏网
    "double-descent-fig1": ("https://arxiv.org/pdf/1912.02292", 1, (105, 388, 527, 506), 220),                          # Nakkiran 等：测试误差随宽度先降后升再降
    # DL 05 CNN
    "lenet5-fig2": ("http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf", 7, (45, 50, 580, 210), 220),                  # LeNet-5 结构（扫描件）
    "alexnet-fig2": ("https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf", 5, (95, 55, 535, 208), 220),  # 原图顶部本来就被裁掉了
    "vgg-table1": ("https://arxiv.org/pdf/1409.1556", 3, (145, 125, 470, 455), 220),                                    # A–E 五种配置
    "resnet-fig2": ("https://arxiv.org/pdf/1512.03385", 2, (95, 70, 265, 158), 260),                                    # 残差块
    "resnet-fig5": ("https://arxiv.org/pdf/1512.03385", 6, (300, 55, 575, 155), 260),                                   # 基本块 vs bottleneck
    "vit-fig1": ("https://arxiv.org/pdf/2010.11929", 3, (100, 65, 510, 245), 220),                                      # ViT 总览
    # DL 06 RNN / attention
    "graves-fig1": ("https://arxiv.org/pdf/1308.0850", 3, (150, 115, 475, 350), 220),                                   # 深层 RNN 沿时间展开
    "graves-fig2": ("https://arxiv.org/pdf/1308.0850", 5, (150, 105, 460, 350), 220),                                   # LSTM 细胞
    "seq2seq-fig1": ("https://arxiv.org/pdf/1409.3215", 2, (95, 315, 520, 405), 220),                                   # 编码器–解码器
    "bahdanau-fig1": ("https://arxiv.org/pdf/1409.0473", 3, (375, 340, 525, 520), 260),                                 # attention 结构
    "bahdanau-fig3a": ("https://arxiv.org/pdf/1409.0473", 6, (105, 82, 302, 278), 260),                                # 对齐矩阵
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
