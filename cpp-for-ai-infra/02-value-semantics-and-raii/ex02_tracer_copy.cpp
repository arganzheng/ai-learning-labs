#include <cstdio>
#include <string>
struct Tracer {
  std::string name;
  Tracer(std::string n) : name(n) { std::printf("ctor      %s\n", name.c_str()); }
  ~Tracer()                       { std::printf("dtor      %s\n", name.c_str()); }
  Tracer(const Tracer& o) : name(o.name + "'") { std::printf("copy-ctor %s\n", name.c_str()); }
};
void by_value(Tracer t) { std::printf("  in by_value, got %s\n", t.name.c_str()); }
void by_cref(const Tracer& t) { std::printf("  in by_cref, got %s\n", t.name.c_str()); }
int main() {
  Tracer a("a");
  Tracer b = a;
  by_value(b);
  by_cref(b);
  std::printf("end of main\n");
}
