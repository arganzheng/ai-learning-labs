// main.cpp
#include <cstdio>
#include "minic10/core/Tensor.h"

using minic10::Tensor;

void print_use(const char* tag, const Tensor& t) {
  std::printf("%s use_count=%u\n", tag, t.use_count());
}

Tensor make_and_fill() {
  Tensor t = minic10::empty({2, 3}, minic10::ScalarType::Float);   // 1 次 malloc
  float* p = t.data_ptr<float>();
  for (int64_t i = 0; i < t.numel(); ++i) p[i] = static_cast<float>(i);
  return t;   // NRVO / 移动：不拷贝 TensorImpl，更不拷贝数据
}

int main() {
  std::printf("== 1. 创建 x ==\n");
  Tensor x = make_and_fill();
  print_use("x", x);

  std::printf("== 2. Tensor y = x ==\n");
  Tensor y = x;                       // 拷贝句柄：refcount 1 -> 2
  print_use("x", x);
  std::printf("same impl? %s, y[4]=%g\n", x.is_same(y) ? "yes" : "no", y.data_ptr<float>()[4]);

  {
    std::printf("== 3. 内层作用域再拷一份 z ==\n");
    Tensor z = y;                     // 2 -> 3
    print_use("x", x);
    std::printf("== 3'. z 离开作用域 ==\n");
  }                                   // ~Tensor(z): 3 -> 2，没有任何释放
  print_use("x", x);

  std::printf("== 4. x = Tensor() ==\n");
  x = Tensor();                       // 2 -> 1：y 还活着，数据不能释放
  print_use("y", y);

  std::printf("== 5. Tensor w = std::move(y) ==\n");
  Tensor w = std::move(y);            // 移动：计数不变，y 变成 undefined
  std::printf("y.defined()=%d, w use_count=%u\n", y.defined(), w.use_count());

  std::printf("== 6. 最后一个句柄 w 离开 main ==\n");
  return 0;
}                                     // ~Tensor(w): 1 -> 0 -> ~TensorImpl -> ~StorageImpl -> ~DataPtr -> free
