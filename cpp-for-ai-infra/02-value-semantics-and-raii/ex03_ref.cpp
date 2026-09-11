#include <cstdio>
int main() {
  int a = 1;
  int& r = a;
  int* p = &a;
  r = 2;
  std::printf("a=%d r=%d *p=%d\n", a, r, *p);
  *p = 3;
  std::printf("a=%d r=%d *p=%d\n", a, r, *p);
  std::printf("&a=%p &r=%p p=%p\n", (void*)&a, (void*)&r, (void*)p);
  int b = 10;
  r = b;   // 不是改绑，是把 b 的值赋给 a
  std::printf("a=%d b=%d &r==&a? %d\n", a, b, &r == &a);
}
