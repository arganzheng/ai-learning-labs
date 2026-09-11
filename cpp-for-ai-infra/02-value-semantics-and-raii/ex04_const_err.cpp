struct Counter {
  int n = 0;
  int get() { return n; }          // 忘了加 const
  void inc() { n++; }
};
int read(const Counter& c) {
  return c.get();                  // 错误
}
int main() {
  const int a = 1;
  a = 2;                           // 错误
  int x = 1;
  const int* p = &x;
  *p = 2;                          // 错误
  int y = 2;
  p = &y;                          // OK
  int* const q = &x;
  *q = 5;                          // OK
  q = &y;                          // 错误
  return 0;
}
