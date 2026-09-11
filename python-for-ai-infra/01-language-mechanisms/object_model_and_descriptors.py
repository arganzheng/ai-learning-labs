"""Python 在 AI-Infra（01）：对象模型 —— type、描述符优先级、bound method、__getattr__ / __getattribute__、__new__、MRO、__hash__、__slots__。
https://arganzheng.life/python-language-mechanisms-and-runtime-internals.html
"""
print("=== class is object / type")
class Runner:
    def run(self, batch):
        return batch
print(type(Runner), Runner.__class__, type(type))
print(Runner.__mro__)
print(sorted(k for k in Runner.__dict__ if not k.startswith("__")), type(Runner.__dict__))
print(type(Runner.__dict__["run"]))

print("=== type() 3-arg")
R2 = type("R2", (object,), {"run": lambda self, b: b})
print(R2, R2().run(1))

print("=== instance dict vs class dict")
class Demo:
    value = 10
obj = Demo()
print(obj.__dict__)
obj.value = 20
print(obj.__dict__, Demo.__dict__["value"], obj.value, Demo.value)

print("=== data descriptor beats instance dict")
class D:
    @property
    def value(self):
        return "from property"
d = D()
d.__dict__["value"] = "from instance dict"
print(d.value, d.__dict__)

print("=== non-data descriptor loses to instance dict")
class Runner2:
    def run(self):
        return "method"
r = Runner2()
r.__dict__["run"] = lambda: "instance attr"
print(r.run())

print("=== function descriptor / bound method")
class Runner3:
    def run(self, batch):
        return batch
r3 = Runner3()
print(Runner3.run, type(Runner3.run))
print(r3.run, type(r3.run))
print(r3.run.__self__ is r3, r3.run.__func__ is Runner3.run)
print(Runner3.__dict__["run"].__get__(r3, Runner3))
print(hasattr(Runner3.__dict__["run"], "__get__"), hasattr(Runner3.__dict__["run"], "__set__"))

print("=== classmethod/staticmethod/property raw objects")
class K:
    @classmethod
    def c(cls): return cls
    @staticmethod
    def s(): return "s"
    @property
    def p(self): return "p"
print(type(K.__dict__["c"]), type(K.__dict__["s"]), type(K.__dict__["p"]))
print(K.__dict__["c"].__get__(None, K), K.__dict__["s"].__get__(None, K))
class Sub(K): pass
print(Sub.c(), Sub().c())
print(hasattr(property, "__set__"), hasattr(classmethod, "__set__"), hasattr(staticmethod, "__set__"))

print("=== __getattr__ only on failure")
class Cfg:
    def __init__(self): self.a = 1
    def __getattr__(self, name):
        print("  __getattr__ called for", name)
        return None
c = Cfg()
print(c.a); print(c.b)

print("=== __getattribute__ on everything")
class Dbg:
    x = 1
    def __getattribute__(self, name):
        print("  __getattribute__:", name)
        return object.__getattribute__(self, name)
Dbg().x

print("=== AttributeError fallback")
class P:
    @property
    def v(self):
        return self.missing
    def __getattr__(self, name):
        return f"fallback({name})"
print(P().v)

print("=== __new__/__init__")
class E:
    def __new__(cls, *a, **k):
        print("  __new__", cls.__name__)
        return super().__new__(cls)
    def __init__(self, v):
        print("  __init__", v)
        self.v = v
E(1)
print(type(E).__call__ is type.__call__)

print("=== MRO")
class A:
    def run(self): return ["A"]
class B:
    def run(self): return ["B"]
class C(A, B): pass
print([k.__name__ for k in C.__mro__], C().run())

class Base:
    def run(self): return ["base"]
class Logging:
    def run(self):
        r = super().run(); r.append("logging"); return r
class Metrics:
    def run(self):
        r = super().run(); r.append("metrics"); return r
class Run(Logging, Metrics, Base): pass
print([k.__name__ for k in Run.__mro__], Run().run())
try:
    class X(A, C): pass
except TypeError as e:
    print("TypeError:", e)

print("=== __eq__ kills __hash__")
class User:
    def __init__(self, uid): self.uid = uid
    def __eq__(self, o): return isinstance(o, User) and self.uid == o.uid
print(User.__hash__)
try:
    {User(1)}
except TypeError as e:
    print("TypeError:", e)

print("=== bool fallback")
class Q:
    def __init__(self, items): self.items = items
    def __len__(self): return len(self.items)
print(bool(Q([])), bool(Q([1])))

print("=== special method looked up on type")
class Callable_:
    def __call__(self): return "type-level"
cobj = Callable_()
cobj.__call__ = lambda: "instance-level"
print(cobj(), cobj.__call__())

print("=== __slots__")
class S:
    __slots__ = ("a",)
s = S(); s.a = 1
print(type(S.__dict__["a"]), hasattr(s, "__dict__"))
