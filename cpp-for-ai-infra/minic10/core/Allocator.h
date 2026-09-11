// minic10/core/Allocator.h
#pragma once
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <memory>

namespace minic10 {

using DeleterFnPtr = void (*)(void*);

// DataPtr：一块"带删除器"的裸内存的唯一所有者。
// 谁分配的、怎么还回去，由 deleter 决定；DataPtr 自己不关心。
class DataPtr {
  std::unique_ptr<void, DeleterFnPtr> ptr_;

  static void deleteNothing(void*) {}

 public:
  DataPtr() : ptr_(nullptr, &deleteNothing) {}
  DataPtr(void* data, DeleterFnPtr deleter)
      : ptr_(data, deleter ? deleter : &deleteNothing) {}

  // unique_ptr 成员已经把"只能移动、不能拷贝"传染给了 DataPtr：Rule of Zero
  void* get() const noexcept { return ptr_.get(); }
  explicit operator bool() const noexcept { return static_cast<bool>(ptr_); }
  void clear() { ptr_.reset(); }
  DeleterFnPtr get_deleter() const noexcept { return ptr_.get_deleter(); }
};

struct Allocator {
  virtual ~Allocator() = default;
  virtual DataPtr allocate(size_t nbytes) = 0;
  virtual DeleterFnPtr raw_deleter() const = 0;
};

struct CPUAllocator final : Allocator {
  static void Delete(void* p) {
    std::printf("  [CPUAllocator] free %p\n", p);
    std::free(p);
  }
  DataPtr allocate(size_t nbytes) override {
    void* p = nbytes == 0 ? nullptr : std::malloc(nbytes);
    std::printf("  [CPUAllocator] malloc %zu bytes -> %p\n", nbytes, p);
    return DataPtr(p, &Delete);
  }
  DeleterFnPtr raw_deleter() const override { return &Delete; }
};

inline Allocator* GetCPUAllocator() {
  static CPUAllocator allocator;   // 生命周期与进程相同；Allocator 从不被 Tensor 拥有
  return &allocator;
}

}  // namespace minic10
