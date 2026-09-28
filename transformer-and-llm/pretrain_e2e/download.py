"""断点续传下载（Range 请求，追加写），网络差、连接常断时比 curl --retry 稳。

    python download.py URL DEST
"""
import os
import sys
import time
import urllib.request


def download(url, dest, chunk=1 << 16, max_retries=1000):
    part = dest + ".part"
    total = None
    for attempt in range(max_retries):
        have = os.path.getsize(part) if os.path.exists(part) else 0
        req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                if r.status == 200 and have:          # 服务器不支持 Range：从头来
                    have = 0; mode = "wb"
                else:
                    mode = "ab"
                if total is None:
                    cr = r.headers.get("Content-Range")
                    total = int(cr.split("/")[-1]) if cr else (int(r.headers.get("Content-Length", 0)) + have)
                with open(part, mode) as f:
                    while True:
                        b = r.read(chunk)
                        if not b:
                            break
                        f.write(b); have += len(b)
            if total and have >= total:
                os.rename(part, dest)
                print(f"完成 {dest}（{have/1e6:.1f} MB）")
                return dest
        except Exception as e:  # noqa: BLE001
            time.sleep(min(30, 2 + attempt))
            if attempt % 10 == 0:
                print(f"  重试 {attempt}：{e}（已下 {have/1e6:.1f} / {(total or 0)/1e6:.1f} MB）")
    raise RuntimeError(f"下载失败 {url}")


if __name__ == "__main__":
    download(sys.argv[1], sys.argv[2])
