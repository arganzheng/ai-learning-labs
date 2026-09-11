#include <cstdio>
#include <vector>
#include <utility>
struct Good {
  Good() = default;
  Good(const Good&) { std::printf("Good copy\n"); }
  Good(Good&&) noexcept { std::printf("Good move\n"); }
};
struct Bad {
  Bad() = default;
  Bad(const Bad&) { std::printf("Bad copy\n"); }
  Bad(Bad&&) { std::printf("Bad move\n"); }   // 没有 noexcept
};
int main() {
  std::vector<Good> g; g.reserve(2); g.emplace_back(); g.emplace_back();
  std::printf("-- Good: push 3rd, capacity %zu -> grow\n", g.capacity());
  g.emplace_back();
  std::vector<Bad> b; b.reserve(2); b.emplace_back(); b.emplace_back();
  std::printf("-- Bad: push 3rd, capacity %zu -> grow\n", b.capacity());
  b.emplace_back();
}
