// minic10/core/StorageImpl.h
#pragma once
#include <cstdio>
#include <utility>
#include "minic10/core/Allocator.h"
#include "minic10/util/intrusive_ptr.h"

namespace minic10 {

struct StorageImpl : intrusive_ptr_target {
  StorageImpl(size_t nbytes, Allocator* allocator)
      : data_(allocator->allocate(nbytes)), nbytes_(nbytes), allocator_(allocator) {
    std::printf("  StorageImpl(%p) ctor, %zu bytes\n", (void*)this, nbytes_);
  }
  ~StorageImpl() override {
    std::printf("  StorageImpl(%p) dtor -> DataPtr 析构 -> deleter\n", (void*)this);
    // data_ 作为成员在这之后自动析构：这就是显存/内存被归还的时刻
  }
  StorageImpl(const StorageImpl&) = delete;
  StorageImpl& operator=(const StorageImpl&) = delete;

  void* data() const noexcept { return data_.get(); }
  size_t nbytes() const noexcept { return nbytes_; }
  Allocator* allocator() const noexcept { return allocator_; }

 private:
  DataPtr data_;
  size_t nbytes_;
  Allocator* allocator_;   // 非拥有：Allocator 是全局的
};

}  // namespace minic10
