#include <cstdio>
#include <utility>
#include <algorithm>
struct Buffer {
  float* data = nullptr;
  int n = 0;
  Buffer() = default;
  explicit Buffer(int n_) : data(new float[n_]), n(n_) { std::printf("alloc %p\n", (void*)data); }
  ~Buffer() { if (data) std::printf("free  %p\n", (void*)data); delete[] data; }
  Buffer(const Buffer& o) : data(new float[o.n]), n(o.n) {
    std::copy(o.data, o.data + n, data);
    std::printf("copy  %p -> %p\n", (void*)o.data, (void*)data);
  }
  Buffer(Buffer&& o) noexcept : data(o.data), n(o.n) {
    o.data = nullptr; o.n = 0;
    std::printf("move  %p (stolen)\n", (void*)data);
  }
  Buffer& operator=(const Buffer& o) { Buffer tmp(o); swap(tmp); return *this; }
  Buffer& operator=(Buffer&& o) noexcept { Buffer tmp(std::move(o)); swap(tmp); return *this; }
  void swap(Buffer& o) noexcept { std::swap(data, o.data); std::swap(n, o.n); }
};
int main() {
  Buffer a(4);
  Buffer b = a;
  Buffer c = std::move(a);
  std::printf("a.data=%p b.data=%p c.data=%p\n", (void*)a.data, (void*)b.data, (void*)c.data);
  b = c;
  std::printf("after b = c\n");
}
