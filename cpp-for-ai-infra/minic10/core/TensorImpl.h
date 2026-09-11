// minic10/core/TensorImpl.h
#pragma once
#include <cstdio>
#include <vector>
#include "minic10/core/DispatchKey.h"
#include "minic10/core/ScalarType.h"
#include "minic10/core/StorageImpl.h"
#include "minic10/util/intrusive_ptr.h"

namespace minic10 {

struct TensorImpl : intrusive_ptr_target {
  TensorImpl(intrusive_ptr<StorageImpl> storage, std::vector<int64_t> sizes,
             ScalarType dtype, DispatchKey key)
      : storage_(std::move(storage)), sizes_(std::move(sizes)), dtype_(dtype), key_(key) {
    strides_.resize(sizes_.size());
    int64_t s = 1;
    for (size_t i = sizes_.size(); i-- > 0;) {
      strides_[i] = s;
      s *= sizes_[i];
    }
    std::printf("  TensorImpl(%p) ctor\n", (void*)this);
  }
  // 虚析构：第 4 篇解释为什么 TensorImpl 需要而 Tensor 不需要
  ~TensorImpl() override {
    std::printf("  TensorImpl(%p) dtor -> 释放对 StorageImpl 的引用\n", (void*)this);
  }
  TensorImpl(const TensorImpl&) = delete;
  TensorImpl& operator=(const TensorImpl&) = delete;

  const std::vector<int64_t>& sizes() const noexcept { return sizes_; }
  const std::vector<int64_t>& strides() const noexcept { return strides_; }
  ScalarType dtype() const noexcept { return dtype_; }
  DispatchKey key() const noexcept { return key_; }
  int64_t numel() const noexcept {
    int64_t n = 1;
    for (auto s : sizes_) n *= s;
    return n;
  }
  const intrusive_ptr<StorageImpl>& storage() const noexcept { return storage_; }
  void* data() const noexcept { return storage_ ? storage_->data() : nullptr; }

 private:
  intrusive_ptr<StorageImpl> storage_;
  std::vector<int64_t> sizes_;
  std::vector<int64_t> strides_;
  ScalarType dtype_;
  DispatchKey key_;
};

}  // namespace minic10
