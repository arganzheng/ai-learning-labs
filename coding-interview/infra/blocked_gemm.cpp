// 面试手撕代码（19）：矩阵乘法的三种写法 —— 朴素 ijk、循环交换 ikj、分块。量化 cache 的影响。
// https://arganzheng.life/coding-interview-infra-concurrency-and-systems.html
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

using Mat = std::vector<float>;   // 行主序 n×n

// 朴素：C[i][j] += A[i][k] * B[k][j]。内层沿 k 走：B[k][j] 每次跳一行（stride n），cache 不友好。
static void gemm_ijk(const Mat& A, const Mat& B, Mat& C, int n) {
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            float s = 0;
            for (int k = 0; k < n; ++k) s += A[i * n + k] * B[k * n + j];
            C[i * n + j] = s;
        }
}

// 循环交换：内层沿 j 走，A[i][k] 是标量、B[k][j] 与 C[i][j] 都连续 → 可向量化。
static void gemm_ikj(const Mat& A, const Mat& B, Mat& C, int n) {
    std::fill(C.begin(), C.end(), 0.f);
    for (int i = 0; i < n; ++i)
        for (int k = 0; k < n; ++k) {
            float a = A[i * n + k];
            for (int j = 0; j < n; ++j) C[i * n + j] += a * B[k * n + j];
        }
}

// 分块：让 A、B、C 的一个 bs×bs 子块同时驻留 cache，块内做 ikj。
static void gemm_blocked(const Mat& A, const Mat& B, Mat& C, int n, int bs) {
    std::fill(C.begin(), C.end(), 0.f);
    for (int i0 = 0; i0 < n; i0 += bs)
        for (int k0 = 0; k0 < n; k0 += bs)
            for (int j0 = 0; j0 < n; j0 += bs)
                for (int i = i0; i < std::min(i0 + bs, n); ++i)
                    for (int k = k0; k < std::min(k0 + bs, n); ++k) {
                        float a = A[i * n + k];
                        for (int j = j0; j < std::min(j0 + bs, n); ++j) C[i * n + j] += a * B[k * n + j];
                    }
}

template <class F> static double time_ms(F f, int reps) {
    auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < reps; ++r) f();
    auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count() / reps;
}

int main(int argc, char** argv) {
    int n = argc > 1 ? std::atoi(argv[1]) : 512;
    Mat A(n * n), B(n * n), C1(n * n), C2(n * n), C3(n * n);
    for (int i = 0; i < n * n; ++i) { A[i] = (i % 7) * 0.5f; B[i] = (i % 5) * 0.25f; }

    double t1 = time_ms([&] { gemm_ijk(A, B, C1, n); }, 1);
    double t2 = time_ms([&] { gemm_ikj(A, B, C2, n); }, 3);
    double t3 = time_ms([&] { gemm_blocked(A, B, C3, n, 64); }, 3);

    float maxdiff = 0;
    for (int i = 0; i < n * n; ++i) maxdiff = std::max(maxdiff, std::max(std::fabs(C1[i] - C2[i]), std::fabs(C1[i] - C3[i])));
    double flops = 2.0 * n * n * n;
    std::printf("n=%d  FLOPs=%.2e  max|diff|=%.1e\n", n, flops, maxdiff);
    std::printf("  ijk      : %8.1f ms  %6.2f GFLOP/s\n", t1, flops / t1 / 1e6);
    std::printf("  ikj      : %8.1f ms  %6.2f GFLOP/s  (%.1fx)\n", t2, flops / t2 / 1e6, t1 / t2);
    std::printf("  blocked64: %8.1f ms  %6.2f GFLOP/s  (%.1fx)\n", t3, flops / t3 / 1e6, t1 / t3);
    return maxdiff < 1e-2 ? 0 : 1;
}
