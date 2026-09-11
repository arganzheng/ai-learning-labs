#include <cstdio>
struct Buffer {
  float* data;
  int n;
  Buffer(int n_) : data(new float[n_]), n(n_) { std::printf("alloc %p\n", (void*)data); std::fflush(stdout); }
  ~Buffer() { std::printf("free  %p\n", (void*)data); std::fflush(stdout); delete[] data; }
};
int main() {
  Buffer a(4);
  Buffer b = a;        // 编译器生成的拷贝构造：逐成员拷贝，b.data == a.data
  std::printf("a.data=%p b.data=%p\n", (void*)a.data, (void*)b.data); std::fflush(stdout);
}
