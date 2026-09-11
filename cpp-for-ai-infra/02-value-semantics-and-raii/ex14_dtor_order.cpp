#include <cstdio>
struct Tag { const char* n; Tag(const char* s) : n(s) { std::printf("ctor %s\n", n); } ~Tag() { std::printf("dtor %s\n", n); } };
struct Base { Tag t{"Base::t"}; ~Base() { std::printf("~Base body\n"); } };
struct Derived : Base { Tag a{"Derived::a"}; Tag b{"Derived::b"}; ~Derived() { std::printf("~Derived body\n"); } };
int main() { std::printf("== enter\n"); { Derived d; std::printf("== leave\n"); } std::printf("== after\n"); }
