// minic10/util/intrusive_ptr.h
#pragma once
#include <cstddef>
#include <cstdint>
#include <utility>

namespace minic10 {

// 被 intrusive_ptr 管理的对象必须继承它：引用计数就住在对象里。
class intrusive_ptr_target {
  template <class T> friend class intrusive_ptr;

  // 第 6 篇会把它改成 std::atomic<uint32_t>，并讨论内存序。
  mutable uint32_t refcount_ = 0;

 protected:
  // 析构函数是 protected + virtual：不允许外界 delete 一个 intrusive_ptr_target*，
  // 但允许 intrusive_ptr 通过基类指针 delete 派生类对象。
  virtual ~intrusive_ptr_target() = default;
  constexpr intrusive_ptr_target() noexcept = default;

  // 拷贝/移动不带走引用计数：计数是"这块内存"的属性，不是"这个值"的属性。
  intrusive_ptr_target(const intrusive_ptr_target&) noexcept : refcount_(0) {}
  intrusive_ptr_target& operator=(const intrusive_ptr_target&) noexcept { return *this; }
};

template <class T>
class intrusive_ptr final {
  T* target_ = nullptr;

  void retain_() noexcept {
    if (target_) ++target_->refcount_;
  }
  void reset_() noexcept {
    if (target_ && --target_->refcount_ == 0) {
      delete target_;   // 通过 T* 删除；~intrusive_ptr_target 是虚的，派生类析构会被调用
    }
    target_ = nullptr;
  }

  // 私有：只允许 make_intrusive / reclaim 从裸指针构造
  struct DontIncreaseRefcount {};
  intrusive_ptr(T* target, DontIncreaseRefcount) noexcept : target_(target) {}

 public:
  using element_type = T;

  intrusive_ptr() noexcept = default;
  /* implicit */ intrusive_ptr(std::nullptr_t) noexcept {}

  // 六大特殊成员函数中的四个：拷贝构造、移动构造、拷贝赋值、移动赋值
  intrusive_ptr(const intrusive_ptr& rhs) noexcept : target_(rhs.target_) { retain_(); }
  intrusive_ptr(intrusive_ptr&& rhs) noexcept : target_(rhs.target_) { rhs.target_ = nullptr; }
  intrusive_ptr& operator=(const intrusive_ptr& rhs) noexcept {
    intrusive_ptr tmp(rhs);   // copy-and-swap：先拿到新引用，再释放旧引用，天然处理自赋值
    swap(tmp);
    return *this;
  }
  intrusive_ptr& operator=(intrusive_ptr&& rhs) noexcept {
    intrusive_ptr tmp(std::move(rhs));
    swap(tmp);
    return *this;
  }
  ~intrusive_ptr() noexcept { reset_(); }

  T* get() const noexcept { return target_; }
  T& operator*() const noexcept { return *target_; }
  T* operator->() const noexcept { return target_; }
  explicit operator bool() const noexcept { return target_ != nullptr; }
  bool defined() const noexcept { return target_ != nullptr; }
  uint32_t use_count() const noexcept { return target_ ? target_->refcount_ : 0; }
  void reset() noexcept { reset_(); }
  void swap(intrusive_ptr& rhs) noexcept { std::swap(target_, rhs.target_); }

  // 与裸指针互转：release() 交出所有权（不减计数），reclaim() 接回所有权（不加计数）。
  // 第 7 篇的 Python 绑定会用到这一对。
  T* release() noexcept {
    T* r = target_;
    target_ = nullptr;
    return r;
  }
  static intrusive_ptr reclaim(T* owning_ptr) noexcept {
    return intrusive_ptr(owning_ptr, DontIncreaseRefcount{});
  }

  template <class... Args>
  static intrusive_ptr make(Args&&... args) {
    intrusive_ptr p(new T(std::forward<Args>(args)...), DontIncreaseRefcount{});
    p.target_->refcount_ = 1;   // 新对象没人能看到，直接写 1，不用原子加
    return p;
  }
};

template <class T, class... Args>
inline intrusive_ptr<T> make_intrusive(Args&&... args) {
  return intrusive_ptr<T>::make(std::forward<Args>(args)...);
}

template <class T>
inline bool operator==(const intrusive_ptr<T>& a, const intrusive_ptr<T>& b) noexcept {
  return a.get() == b.get();
}

}  // namespace minic10
