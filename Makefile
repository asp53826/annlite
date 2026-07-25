# One clang++ invocation, no cmake needed.
CXX      ?= c++
PYTHON   ?= .venv/bin/python
SDK      := $(shell xcrun --show-sdk-path 2>/dev/null)
CXXFLAGS := -O3 -std=c++17 -Wall -Wextra -fPIC -march=native -Iinclude

# Some Command Line Tools installs ship a truncated libc++ header dir that
# clang prefers over the SDK's complete one, so <vector> stops resolving.
# Point at the SDK copy explicitly when that's the case.
ifneq ($(SDK),)
ifeq ($(shell echo '\#include <vector>' | $(CXX) -std=c++17 -fsyntax-only -x c++ - 2>/dev/null && echo ok),)
CXXFLAGS += -isystem $(SDK)/usr/include/c++/v1
endif
endif

SUFFIX := $(shell $(PYTHON)-config --extension-suffix 2>/dev/null || echo .so)
PYINC  := $(shell $(PYTHON) -c "import pybind11,sysconfig;print('-I'+pybind11.get_include()+' -I'+sysconfig.get_paths()['include'])")
UNAME  := $(shell uname -s)
ifeq ($(UNAME),Darwin)
LDFLAGS := -shared -undefined dynamic_lookup
else
LDFLAGS := -shared
endif

MODULE := python/_annlite$(SUFFIX)

all: $(MODULE)

$(MODULE): python/bindings.cpp src/hnsw.cpp include/annlite/*.hpp
	$(CXX) $(CXXFLAGS) $(PYINC) $(LDFLAGS) python/bindings.cpp src/hnsw.cpp -o $@

test: all
	$(PYTHON) -m pytest tests -q

bench: all
	$(PYTHON) bench/benchmark.py

clean:
	rm -f python/_annlite*.so

.PHONY: all test bench clean
