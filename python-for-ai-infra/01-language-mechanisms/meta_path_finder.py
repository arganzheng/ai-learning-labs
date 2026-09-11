"""Python 在 AI-Infra（01）：自定义 MetaPathFinder —— 屏蔽一个模块、追踪 import 请求。
https://arganzheng.life/python-language-mechanisms-and-runtime-internals.html
"""
import sys, importlib.abc, importlib.util
class Blocker(importlib.abc.MetaPathFinder):
    def __init__(self, names): self.names = names
    def find_spec(self, fullname, path, target=None):
        if fullname in self.names:
            raise ModuleNotFoundError(f"{fullname} blocked for testing", name=fullname)
        return None
sys.meta_path.insert(0, Blocker({"numpy"}))
print(importlib.util.find_spec("json") is not None)
try:
    import numpy
except ModuleNotFoundError as e:
    print("ModuleNotFoundError:", e)

class Tracer(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        print(f"  finder asked for {fullname!r}, path={None if path is None else [p.split('/')[-1] for p in path]}")
        return None
sys.meta_path.insert(0, Tracer())
import xml.dom.minidom
