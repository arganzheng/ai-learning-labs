#include <cstdio>
#include "minic10/util/intrusive_ptr.h"
using namespace minic10;

struct Widget : intrusive_ptr_target {
  int id;
  explicit Widget(int i) : id(i) { std::printf("Widget(%d) ctor\n", id); }
  ~Widget() override { std::printf("Widget(%d) dtor\n", id); }
};

int main() {
  std::printf("sizeof(intrusive_ptr<Widget>)=%zu sizeof(Widget*)=%zu\n",
              sizeof(intrusive_ptr<Widget>), sizeof(Widget*));
  intrusive_ptr<Widget> a = make_intrusive<Widget>(1);
  std::printf("a.use_count=%u\n", a.use_count());
  {
    intrusive_ptr<Widget> b = a;                 // 拷贝：+1
    std::printf("after copy: use_count=%u, same? %d\n", a.use_count(), a == b);
    intrusive_ptr<Widget> c = std::move(b);      // 移动：计数不变，b 变空
    std::printf("after move: use_count=%u, b.defined=%d\n", a.use_count(), b.defined());
  }                                              // c 析构：-1
  std::printf("after scope: use_count=%u\n", a.use_count());

  Widget* raw = a.release();                     // 交出所有权：计数不变，a 变空
  std::printf("after release: a.defined=%d raw->id=%d\n", a.defined(), raw->id);
  a = intrusive_ptr<Widget>::reclaim(raw);       // 接回所有权：计数不变
  std::printf("after reclaim: use_count=%u\n", a.use_count());
  a = nullptr;                                   // 最后一个引用：-1 -> 0 -> delete
  std::printf("end\n");
}
