// 面试手撕代码（19）：C++ 版 Top-K（priority_queue / nth_element）与线程安全 LRU（unordered_map + list + mutex）。
// https://arganzheng.life/coding-interview-infra-concurrency-and-systems.html
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <list>
#include <mutex>
#include <optional>
#include <queue>
#include <thread>
#include <unordered_map>
#include <vector>

// 前 k 大：大小为 k 的最小堆。O(n log k)。
std::vector<int> topk_heap(const std::vector<int>& a, int k) {
    std::priority_queue<int, std::vector<int>, std::greater<int>> pq;   // 最小堆
    for (int x : a) {
        if ((int)pq.size() < k) pq.push(x);
        else if (x > pq.top()) { pq.pop(); pq.push(x); }
    }
    std::vector<int> out;
    while (!pq.empty()) { out.push_back(pq.top()); pq.pop(); }
    return out;                                                    // 升序
}

// 第 k 大：nth_element 是 introselect，期望 O(n)，原地。
int kth_largest(std::vector<int> a, int k) {
    std::nth_element(a.begin(), a.begin() + (k - 1), a.end(), std::greater<int>());
    return a[k - 1];
}

// 线程安全 LRU：unordered_map<key, list::iterator> + list<pair>（list 的 splice 是 O(1) 且不使迭代器失效）。
template <class K, class V>
class LRUCache {
    size_t cap_;
    std::list<std::pair<K, V>> items_;                             // front = 最近使用
    std::unordered_map<K, typename std::list<std::pair<K, V>>::iterator> map_;
    std::mutex mu_;

public:
    explicit LRUCache(size_t cap) : cap_(cap) {}

    std::optional<V> get(const K& k) {
        std::lock_guard<std::mutex> g(mu_);
        auto it = map_.find(k);
        if (it == map_.end()) return std::nullopt;
        items_.splice(items_.begin(), items_, it->second);        // 移到头：O(1)，迭代器仍有效
        return it->second->second;
    }

    void put(const K& k, V v) {
        std::lock_guard<std::mutex> g(mu_);
        auto it = map_.find(k);
        if (it != map_.end()) {
            it->second->second = std::move(v);
            items_.splice(items_.begin(), items_, it->second);
            return;
        }
        if (map_.size() == cap_) {
            map_.erase(items_.back().first);
            items_.pop_back();
        }
        items_.emplace_front(k, std::move(v));
        map_[k] = items_.begin();
    }

    size_t size() { std::lock_guard<std::mutex> g(mu_); return map_.size(); }
};

int main() {
    std::vector<int> a = {3, 2, 1, 5, 6, 4};
    assert((topk_heap(a, 2) == std::vector<int>{5, 6}));
    assert(kth_largest(a, 2) == 5 && kth_largest(a, 6) == 1);

    LRUCache<int, int> c(2);
    c.put(1, 1); c.put(2, 2);
    assert(c.get(1) == 1);
    c.put(3, 3);
    assert(!c.get(2).has_value());
    c.put(4, 4);
    assert(!c.get(1).has_value() && c.get(3) == 3 && c.get(4) == 4);

    // 多线程读写：不崩、大小不超容量
    LRUCache<int, int> shared(64);
    std::vector<std::thread> ts;
    for (int t = 0; t < 8; ++t)
        ts.emplace_back([&shared, t] {
            for (int i = 0; i < 20000; ++i) {
                int k = (i * 7 + t) % 200;
                if (!shared.get(k)) shared.put(k, k * k);
            }
        });
    for (auto& t : ts) t.join();
    assert(shared.size() <= 64);
    std::printf("topk_lru: topk/nth_element OK, LRU OK, 8 threads x 20000 ops, size=%zu <= 64\n", shared.size());
    return 0;
}
