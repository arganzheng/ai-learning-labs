#include <cstdio>
#include <memory>
struct Node { const char* name; explicit Node(const char* n) : name(n) {} ~Node() { std::printf("~Node %s\n", name); } };
int main() {
  std::printf("sizeof: raw=%zu unique=%zu shared=%zu weak=%zu\n",
    sizeof(Node*), sizeof(std::unique_ptr<Node>), sizeof(std::shared_ptr<Node>), sizeof(std::weak_ptr<Node>));
  auto sp1 = std::make_shared<Node>("n1");
  std::printf("use_count=%ld\n", sp1.use_count());
  { auto sp2 = sp1; std::printf("use_count=%ld\n", sp1.use_count()); }
  std::printf("use_count=%ld\n", sp1.use_count());
  std::weak_ptr<Node> wp = sp1;
  std::printf("weak: expired=%d use_count=%ld\n", wp.expired(), wp.use_count());
  if (auto locked = wp.lock()) std::printf("locked -> %s, use_count=%ld\n", locked->name, locked.use_count());
  sp1.reset();
  std::printf("after reset: expired=%d, lock()==nullptr? %d\n", wp.expired(), wp.lock() == nullptr);
}
