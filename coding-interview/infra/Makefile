# 面试手撕代码（19）Infra 岗手撕。  make        编译三个 C++ 程序    make run    跑 Python + C++ 全部自检
CXX      ?= c++
CXXFLAGS ?= -std=c++17 -Wall -Wextra -O2
BUILD    := build
SRCS     := $(wildcard *.cpp)
BINS     := $(patsubst %.cpp,$(BUILD)/%,$(SRCS))
PYTHON   ?= python3

.PHONY: all run clean

all: $(BINS)

$(BUILD)/%: %.cpp | $(BUILD)
	$(CXX) $(CXXFLAGS) -pthread -o $@ $<

$(BUILD):
	mkdir -p $(BUILD)

run: all
	$(PYTHON) concurrency.py
	$(BUILD)/memory_pool
	$(BUILD)/topk_lru
	$(BUILD)/blocked_gemm 512

clean:
	rm -rf $(BUILD)
