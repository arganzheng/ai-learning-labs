#include <cstdio>
#include <cstdlib>
#include <memory>
void my_free(void* p) { std::printf("my_free(%p)\n", p); std::free(p); }
int main() {
  std::unique_ptr<void, void(*)(void*)> p(std::malloc(16), &my_free);
  std::printf("sizeof unique_ptr<void, fnptr>=%zu  get()=%p\n", sizeof(p), p.get());
  std::unique_ptr<int> q(new int(7));
  std::unique_ptr<int> r = std::move(q);
  std::printf("q==nullptr? %d  *r=%d\n", q == nullptr, *r);
}
