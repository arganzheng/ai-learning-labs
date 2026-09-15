// 面试手撕代码（19）：固定大小块的内存池 —— 空闲链表 O(1) 分配 / 释放。
// https://arganzheng.life/coding-interview-infra-concurrency-and-systems.html
#include <cassert>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <vector>

class FixedPool {
    // 每个空闲块的前 8 字节复用为"下一个空闲块"的指针：不需要额外的元数据数组。
    struct Node { Node* next; };
    char* buf_;
    Node* free_;
    size_t block_, count_, in_use_ = 0;

public:
    FixedPool(size_t block_size, size_t count)
        : block_(block_size < sizeof(Node) ? sizeof(Node) : block_size), count_(count) {
        buf_ = static_cast<char*>(std::malloc(block_ * count_));
        if (!buf_) throw std::bad_alloc();
        free_ = nullptr;
        for (size_t i = count_; i-- > 0;) {                 // 倒着串：分配时地址递增，对 cache 友好
            Node* n = reinterpret_cast<Node*>(buf_ + i * block_);
            n->next = free_;
            free_ = n;
        }
    }
    ~FixedPool() { std::free(buf_); }
    FixedPool(const FixedPool&) = delete;
    FixedPool& operator=(const FixedPool&) = delete;

    void* alloc() {
        if (!free_) return nullptr;                         // 池满：返回空（或扩容一个新 chunk）
        Node* n = free_;
        free_ = n->next;
        ++in_use_;
        return n;
    }
    void release(void* p) {
        assert(owns(p));
        Node* n = static_cast<Node*>(p);
        n->next = free_;
        free_ = n;
        --in_use_;
    }
    bool owns(const void* p) const {
        const char* c = static_cast<const char*>(p);
        return c >= buf_ && c < buf_ + block_ * count_ && (c - buf_) % block_ == 0;
    }
    size_t in_use() const { return in_use_; }
    size_t capacity() const { return count_; }
};

int main() {
    FixedPool pool(64, 4);
    void* a = pool.alloc();
    void* b = pool.alloc();
    void* c = pool.alloc();
    void* d = pool.alloc();
    assert(a && b && c && d && pool.alloc() == nullptr);  // 第五次分配失败
    assert(static_cast<char*>(b) - static_cast<char*>(a) == 64);
    pool.release(b);
    void* e = pool.alloc();
    assert(e == b);                                       // 刚释放的块被复用（LIFO）
    assert(pool.in_use() == 4 && pool.owns(e) && !pool.owns(&pool));
    pool.release(a); pool.release(c); pool.release(d); pool.release(e);
    assert(pool.in_use() == 0);
    std::printf("memory_pool: 4 blocks x 64B, alloc/free O(1), LIFO reuse OK\n");
    return 0;
}
