# 面试手撕代码 —— Java 解法。  make test  编译并用 -ea 跑每个类的 main 里的断言。
JAVA_HOME ?= /opt/homebrew/opt/openjdk
JAVAC := $(wildcard $(JAVA_HOME)/bin/javac)
JAVA  := $(wildcard $(JAVA_HOME)/bin/java)
ifeq ($(JAVAC),)
JAVAC := javac
JAVA  := java
endif

SRCS  := $(wildcard P*.java)
BUILD := build

.PHONY: all test clean

all: $(BUILD)/.compiled

$(BUILD)/.compiled: $(SRCS)
	mkdir -p $(BUILD)
	$(JAVAC) --release 21 -Xlint:all -d $(BUILD) $(SRCS)
	touch $@

test: all
	@for f in $(SRCS); do $(JAVA) -ea -cp $(BUILD) $${f%.java} || exit 1; done

clean:
	rm -rf $(BUILD)
