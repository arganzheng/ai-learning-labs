#include <cstdio>
#include <stdexcept>
struct Guard {
  const char* name;
  explicit Guard(const char* n) : name(n) { std::printf("acquire %s\n", name); }
  ~Guard() { std::printf("release %s\n", name); }
  Guard(const Guard&) = delete;
  Guard& operator=(const Guard&) = delete;
};
struct Holder {
  Guard first{"member-1"};
  Guard second{"member-2"};
  ~Holder() { std::printf("~Holder body\n"); }
};
void work(bool fail) {
  Guard g("local");
  if (fail) throw std::runtime_error("boom");
  std::printf("work done normally\n");
}
int main() {
  std::printf("== normal\n");
  work(false);
  std::printf("== exception\n");
  try { work(true); } catch (const std::exception& e) { std::printf("caught: %s\n", e.what()); }
  std::printf("== members\n");
  { Holder h; }
  std::printf("== end\n");
}
