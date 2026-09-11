#include <cstdio>
struct Point { double x, y; };
int main() {
  Point p{1.0, 2.0};
  Point q = p;
  q.x = 3.0;
  Point* hp = &p;
  std::printf("sizeof(Point)=%zu\n", sizeof(Point));
  std::printf("p at %p  q at %p  hp=%p (value of hp is the address of p)\n", (void*)&p, (void*)&q, (void*)hp);
  std::printf("p.x=%g q.x=%g hp->x=%g (*hp).x=%g\n", p.x, q.x, hp->x, (*hp).x);
  Point* heap = new Point{5.0, 6.0};
  std::printf("heap object at %p, pointer variable itself at %p\n", (void*)heap, (void*)&heap);
  delete heap;
}
