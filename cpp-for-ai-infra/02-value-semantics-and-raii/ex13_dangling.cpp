#include <cstdio>
#include <string>
const std::string& bad() {
  std::string local = "hello";
  return local;   // 返回局部变量的引用
}
int main() { const std::string& s = bad(); std::printf("%zu\n", s.size()); }
