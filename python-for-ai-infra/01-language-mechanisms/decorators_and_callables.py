"""Python 在 AI-Infra（01）：装饰器顺序与可调用对象 —— 装饰 classmethod 的 TypeError、类装饰器丢失 self、用 __get__ 修好。
https://arganzheng.life/python-language-mechanisms-and-runtime-internals.html
"""
def log_call(func):
    def wrapper(*a, **k):
        return func(*a, **k)
    return wrapper
try:
    class R:
        @log_call
        @classmethod
        def create(cls): return cls
    R.create()
except TypeError as e:
    print("TypeError:", e)

class LogCall:
    def __init__(self, func): self.func = func
    def __call__(self, *a, **k): return self.func(*a, **k)
class S:
    @LogCall
    def m(self): return "ok"
try:
    S().m()
except TypeError as e:
    print("TypeError:", e)
import functools
class LogCall2(LogCall):
    def __get__(self, obj, objtype=None):
        return self if obj is None else functools.partial(self, obj)
class S2:
    @LogCall2
    def m(self): return "ok"
print(S2().m())
from contextlib import suppress
print(type(suppress(ValueError).__exit__(ValueError, ValueError(), None)))
