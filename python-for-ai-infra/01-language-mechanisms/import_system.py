"""Python 在 AI-Infra（01）：import 系统 —— spec、loader、meta_path、path_hooks、find_spec、sys.modules。
https://arganzheng.life/python-language-mechanisms-and-runtime-internals.html
"""
import sys, json, importlib, importlib.util, importlib.machinery
print("=== json spec")
print(json.__spec__)
print(json.__loader__)
print(json.__package__)
print(json.__file__)
print(json.__path__)
print("--- json.decoder")
import json.decoder
print(json.decoder.__spec__)
print(json.decoder.__package__)

print("=== math (builtin)")
import math
print(math.__spec__)
print(hasattr(math, "__file__"))

print("=== extension module")
import numpy
m = sys.modules["numpy._core._multiarray_umath"]
print(m.__spec__.loader)
print(m.__spec__.origin)

print("=== meta_path")
for f in sys.meta_path:
    print(" ", f)
print("=== path_hooks")
for h in sys.path_hooks:
    print(" ", h)
print("=== suffixes")
print(importlib.machinery.SOURCE_SUFFIXES)
print(importlib.machinery.BYTECODE_SUFFIXES)
print(importlib.machinery.EXTENSION_SUFFIXES)

print("=== find_spec without import")
print("triton" in sys.modules, importlib.util.find_spec("triton"))
spec = importlib.util.find_spec("numpy.linalg")
print(spec.name, spec.origin.split("site-packages/")[-1], "numpy.linalg" in sys.modules)

print("=== path_importer_cache sample")
k = list(sys.path_importer_cache.items())[-1]
print(k[0].split("/")[-1], k[1])

print("=== sys.modules identity")
import json as j2
print(j2 is json, sys.modules["json"] is json)

print("=== module type / dict")
import types
print(type(json), isinstance(json, types.ModuleType))
print(list(json.__dict__)[:6])
