# C++ 在 AI-Infra — build everything with:   make        run the well-behaved examples:   make run
CXX      ?= c++
CXXFLAGS ?= -std=c++17 -Wall -Wextra -O0 -g
BUILD    := build

EX_DIR   := 02-value-semantics-and-raii
# ex04 is meant NOT to compile (const-correctness errors); build it with `make ex04` to read the diagnostics.
EX_SRCS  := $(filter-out $(EX_DIR)/ex04_const_err.cpp,$(wildcard $(EX_DIR)/ex*.cpp))
EX_BINS  := $(patsubst $(EX_DIR)/%.cpp,$(BUILD)/%,$(EX_SRCS))

MINIC10_HDRS := $(wildcard minic10/core/*.h minic10/util/*.h)
MINIC10_BINS := $(BUILD)/minic10_tensor $(BUILD)/minic10_intrusive

# ex05 double-frees on purpose, ex13 returns a dangling reference: they crash or print garbage by design.
RUN_SKIP := $(BUILD)/ex05_buffer_bad $(BUILD)/ex13_dangling

all: $(EX_BINS) $(MINIC10_BINS)

$(BUILD)/%: $(EX_DIR)/%.cpp | $(BUILD)
	$(CXX) $(CXXFLAGS) -o $@ $<

$(BUILD)/minic10_tensor: minic10/main.cpp $(MINIC10_HDRS) | $(BUILD)
	$(CXX) $(CXXFLAGS) -I. -o $@ $<

$(BUILD)/minic10_intrusive: minic10/test_intrusive.cpp $(MINIC10_HDRS) | $(BUILD)
	$(CXX) $(CXXFLAGS) -I. -o $@ $<

$(BUILD):
	mkdir -p $(BUILD)

run: all
	@for b in $(filter-out $(RUN_SKIP),$(EX_BINS)) $(MINIC10_BINS); do echo "===== $$b"; ./$$b; echo; done

ex04:
	-$(CXX) $(CXXFLAGS) -o $(BUILD)/ex04_const_err $(EX_DIR)/ex04_const_err.cpp

clean:
	rm -rf $(BUILD)

.PHONY: all run ex04 clean
