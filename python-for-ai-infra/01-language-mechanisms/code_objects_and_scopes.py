"""Python 在 AI-Infra（01）：函数与代码对象 —— dis、code object、frame、traceback、LEGB、闭包晚绑定、nonlocal、UnboundLocalError、is 与 ==。
https://arganzheng.life/python-language-mechanisms-and-runtime-internals.html
"""
import dis, sys, inspect
print("PY", sys.version)

print("=== dis add")
def add(x, y):
    return x + y
dis.dis(add)

print("=== code object")
c = add.__code__
print(type(c))
print(c.co_varnames, c.co_argcount, c.co_consts, c.co_names)
print(c.co_code[:8])

print("=== function object attrs")
def predict(x, scale=2):
    """Scale the input."""
    return x * scale
print(type(predict))
print(predict.__code__)
print(predict.__defaults__)
print(predict.__doc__)
print(predict.__module__, predict.__qualname__)
print(predict.__globals__ is globals())
predict.tag = "demo"
print(predict.__dict__)

print("=== two functions share nothing but code?")
def make():
    def f():
        return 1
    return f
a, b = make(), make()
print(a is b, a.__code__ is b.__code__)

print("=== frames")
def show_frame():
    frame = inspect.currentframe()
    print(frame.f_code.co_name, list(frame.f_locals), frame.f_back.f_code.co_name, frame.f_lineno)
show_frame()

print("=== traceback frames")
import traceback
def inner():
    raise ValueError("boom")
def outer():
    inner()
try:
    outer()
except ValueError as exc:
    tb = exc.__traceback__
    while tb:
        print(tb.tb_frame.f_code.co_name, tb.tb_lineno)
        tb = tb.tb_next

print("=== LEGB dis: local/global/free")
value = "global"
def outer2():
    value = "enclosing"
    def inner2():
        return value
    return inner2
f = outer2()
dis.dis(f)
print(f.__code__.co_freevars, f.__closure__, f.__closure__[0].cell_contents)
print(outer2.__code__.co_cellvars)

print("=== late binding")
functions = []
for i in range(3):
    functions.append(lambda: i)
print([fn() for fn in functions])
print([fn.__closure__ for fn in functions])
print(functions[0].__code__.co_names, functions[0].__code__.co_freevars)

def make_fns():
    fns = []
    for i in range(3):
        fns.append(lambda: i)
    return fns
fns = make_fns()
print([fn() for fn in fns], [id(fn.__closure__[0]) for fn in fns])

print("=== nonlocal")
def make_counter(start=0):
    count = start
    def increment():
        nonlocal count
        count += 1
        return count
    return increment
counter = make_counter()
print(counter(), counter())
print(counter.__closure__[0].cell_contents)

print("=== UnboundLocalError")
def bad():
    count = 0
    def inc():
        count += 1
        return count
    return inc
try:
    bad()()
except UnboundLocalError as e:
    print(type(e).__name__, e)

print("=== is / == ints")
a = 1000; b = 1000
print(a == b, a is b)
x = 256; y = 256
print(x is y)
p = int("1000"); q = int("1000")
print(p == q, p is q)
