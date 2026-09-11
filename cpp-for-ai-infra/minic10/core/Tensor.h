// minic10/core/Tensor.h
#pragma once
#include <utility>
#include <vector>
#include "minic10/core/Allocator.h"
#include "minic10/core/TensorImpl.h"
#include "minic10/util/intrusive_ptr.h"

namespace minic10 {

// Tensor 是句柄：唯一的数据成员是一个 intrusive_ptr<TensorImpl>。
// 拷贝 Tensor = 拷贝一个指针 + 引用计数 +1；从不拷贝数据。
class Tensor {
  intrusive_ptr<TensorImpl> impl_;

 public:
  Tensor() = default;
  explicit Tensor(intrusive_ptr<TensorImpl> impl) : impl_(std::move(impl)) {}
  // Rule of Zero：拷贝/移动/析构全部交给 impl_ 的 intrusive_ptr 生成，一行都不用写。

  bool defined() const noexcept { return impl_.defined(); }
  uint32_t use_count() const noexcept { return impl_.use_count(); }
  const std::vector<int64_t>& sizes() const { return impl_->sizes(); }
  const std::vector<int64_t>& strides() const { return impl_->strides(); }
  ScalarType dtype() const { return impl_->dtype(); }
  int64_t numel() const { return impl_->numel(); }
  template <typename T>
  T* data_ptr() const { return static_cast<T*>(impl_->data()); }

  TensorImpl* unsafeGetTensorImpl() const noexcept { return impl_.get(); }
  TensorImpl* unsafeReleaseTensorImpl() noexcept { return impl_.release(); }
  bool is_same(const Tensor& other) const noexcept { return impl_ == other.impl_; }
};

// 按值返回：调用方拿到的是移动/RVO 过来的句柄，没有数据拷贝
inline Tensor empty(std::vector<int64_t> sizes, ScalarType dtype) {
  int64_t numel = 1;
  for (auto s : sizes) numel *= s;
  auto storage = make_intrusive<StorageImpl>(numel * itemsize(dtype), GetCPUAllocator());
  return Tensor(make_intrusive<TensorImpl>(std::move(storage), std::move(sizes), dtype,
                                           DispatchKey::CPU));
}

}  // namespace minic10
