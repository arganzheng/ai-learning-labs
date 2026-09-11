// minic10/core/ScalarType.h（第 3 篇会补上到 C++ 类型的映射）
#pragma once
#include <cstddef>
namespace minic10 {
enum class ScalarType { Float, Double, Long };
inline size_t itemsize(ScalarType t) {
  switch (t) {
    case ScalarType::Float: return 4;
    case ScalarType::Double: return 8;
    case ScalarType::Long: return 8;
  }
  return 0;
}
}  // namespace minic10
