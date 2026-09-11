"""Python 在 AI-Infra（01）：协议与控制流 —— property 与 __getattr__ 的坑、super 的 __class__ cell、生成器状态与 close、with 展开、contextmanager、异常链、finally。
https://arganzheng.life/python-language-mechanisms-and-runtime-internals.html
"""
import sys, dis, inspect
print("=== property AttributeError masked by __getattr__")
class P:
    @property
    def v(self):
        raise AttributeError("bug inside property")
    def __getattr__(self, name):
        return f"fallback({name})"
print(P().v)

print("=== super uses __class__ cell")
class Base:
    def run(self): return "base"
class Child(Base):
    def run(self):
        return super().run()
print(Child.run.__code__.co_freevars)

print("=== generator states")
def stream():
    print("  start")
    yield 1
    print("  resume")
    yield 2
    print("  end")
g = stream()
print(type(g), inspect.getgeneratorstate(g))
print("created")
print(next(g), inspect.getgeneratorstate(g))
print("between")
print(next(g))
try:
    next(g)
except StopIteration:
    print("StopIteration", inspect.getgeneratorstate(g))

print("=== generator frame persists")
def gen():
    local = "kept"
    yield local
    yield local
g = gen(); next(g)
print(g.gi_frame.f_locals, g.gi_frame.f_lineno)

print("=== close -> GeneratorExit -> finally")
def reader():
    print("  open")
    try:
        yield 1
        yield 2
    except GeneratorExit:
        print("  GeneratorExit")
        raise
    finally:
        print("  close file")
r = reader(); next(r); r.close()
print("--- break exits for-loop, but generator not closed yet")
r = reader()
for v in r:
    break
print("after break; refcount drop ->")
del r
print("--- yield from return value")
def sub():
    yield 1
    return "sub-result"
def outer():
    result = yield from sub()
    print("  sub returned:", result)
list(outer())

print("=== with expansion")
class Res:
    def __enter__(self):
        print("  enter"); return self
    def __exit__(self, t, v, tb):
        print("  exit", t.__name__ if t else None); return False
try:
    with Res():
        raise ValueError("x")
except ValueError:
    print("propagated")
class Suppress(Res):
    def __exit__(self, t, v, tb):
        print("  suppress", t.__name__); return True
with Suppress():
    raise ValueError("x")
print("suppressed")
class BadEnter:
    def __enter__(self): raise RuntimeError("enter failed")
    def __exit__(self, *a): print("  exit should NOT run")
try:
    with BadEnter(): pass
except RuntimeError as e: print(e)

print("=== contextmanager: exception is thrown into generator at yield")
from contextlib import contextmanager
@contextmanager
def cm():
    print("  acquire")
    try:
        yield "h"
    except ValueError as e:
        print("  saw", e)
        raise
    finally:
        print("  release")
try:
    with cm() as h:
        raise ValueError("boom")
except ValueError:
    print("propagated")

print("=== exception chain")
def load():
    raise OSError("no such file")
try:
    try:
        load()
    except OSError as exc:
        raise RuntimeError("failed to load configuration") from exc
except RuntimeError as e:
    print(repr(e.__cause__), e.__suppress_context__)
try:
    try:
        load()
    except OSError:
        raise RuntimeError("implicit")
except RuntimeError as e:
    print(repr(e.__context__), e.__cause__)

print("=== bare raise keeps traceback")
import traceback
def w():
    raise ValueError("inner")
try:
    try:
        w()
    except ValueError:
        raise
except ValueError as e:
    print([f.name for f in traceback.extract_tb(e.__traceback__)])
try:
    try:
        w()
    except ValueError as e:
        raise e
except ValueError as e:
    print([f.name for f in traceback.extract_tb(e.__traceback__)])

print("=== BaseException tree")
print([c.__name__ for c in BaseException.__subclasses__()])

print("=== finally runs on return")
def f():
    try:
        return "ret"
    finally:
        print("  finally")
print(f())

print("=== iter protocol")
class CountDown:
    def __init__(self, s): self.cur = s
    def __iter__(self): return self
    def __next__(self):
        if self.cur <= 0: raise StopIteration
        self.cur -= 1; return self.cur + 1
c = CountDown(2)
print(list(c), list(c))
print("--- iterable vs iterator: list")
l = [1, 2]
print(iter(l) is l, iter(l) is iter(l))
print("--- getitem fallback iteration")
class Seq:
    def __getitem__(self, i):
        if i >= 3: raise IndexError
        return i * 10
print(list(Seq()), 20 in Seq())
