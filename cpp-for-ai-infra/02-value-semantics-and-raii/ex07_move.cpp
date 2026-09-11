#include <cstdio>
#include <string>
#include <utility>
struct Tracer {
  std::string name;
  Tracer(std::string n) : name(std::move(n)) { std::printf("ctor      %s\n", name.c_str()); }
  ~Tracer()                                    { std::printf("dtor      %s\n", name.c_str()); }
  Tracer(const Tracer& o) : name(o.name + "'") { std::printf("copy-ctor %s\n", name.c_str()); }
  Tracer(Tracer&& o) noexcept : name(std::move(o.name)) {
    o.name = "(moved-from)";
    std::printf("move-ctor %s\n", name.c_str());
  }
};
void take(const Tracer&) { std::printf("  take(const Tracer&)\n"); }
void take(Tracer&&)      { std::printf("  take(Tracer&&)\n"); }
Tracer make() { Tracer t("ret"); return t; }
Tracer make_bad() { Tracer t("ret2"); return std::move(t); }
int main() {
  Tracer a("a");
  std::printf("-- take(a)\n");            take(a);
  std::printf("-- take(Tracer(\"tmp\"))\n"); take(Tracer("tmp"));
  std::printf("-- take(std::move(a))\n"); take(std::move(a));
  std::printf("-- a still: %s\n", a.name.c_str());
  std::printf("-- Tracer b = a\n");       Tracer b = a;
  std::printf("-- Tracer c = std::move(a)\n"); Tracer c = std::move(a);
  std::printf("-- a now: %s\n", a.name.c_str());
  std::printf("-- Tracer d = make()\n");  Tracer d = make();
  std::printf("-- Tracer e = make_bad()\n"); Tracer e = make_bad();
  std::printf("-- end\n");
}
