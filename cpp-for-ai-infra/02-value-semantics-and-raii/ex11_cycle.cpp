#include <cstdio>
#include <memory>
struct B;
struct A { std::shared_ptr<B> b; ~A() { std::printf("~A\n"); } };
struct B { std::shared_ptr<A> a; ~B() { std::printf("~B\n"); } };
struct D;
struct C { std::shared_ptr<D> d; ~C() { std::printf("~C\n"); } };
struct D { std::weak_ptr<C> c;   ~D() { std::printf("~D\n"); } };
int main() {
  {
    auto a = std::make_shared<A>(); auto b = std::make_shared<B>();
    a->b = b; b->a = a;
    std::printf("cycle: a.use_count=%ld b.use_count=%ld\n", a.use_count(), b.use_count());
  }
  std::printf("-- left scope (A/B never destroyed)\n");
  {
    auto c = std::make_shared<C>(); auto d = std::make_shared<D>();
    c->d = d; d->c = c;
    std::printf("weak: c.use_count=%ld d.use_count=%ld\n", c.use_count(), d.use_count());
  }
  std::printf("-- left scope\n");
}
