CXX = c++ -std=c++11
CXXFLAGS = -O3 -Wall -Wextra
MCHEP_DEPS != pkg-config --cflags --libs mchep_capi
MATH_LIBS = -lm

MCHEP_TARGETS = benchmark_mchep benchmark_mchep_simd
CUBA_TARGET = cuba_example
TARGETS = $(MCHEP_TARGETS) $(CUBA_TARGET)

all: $(TARGETS)

# --- MCHEP Benchmarks ---
benchmark_mchep: benchmark_mchep.cpp
	$(CXX) $(CXXFLAGS) $< $(MCHEP_DEPS) $(MATH_LIBS) -o $@

benchmark_mchep_simd: benchmark_mchep_simd.cpp
	$(CXX) $(CXXFLAGS) $< $(MCHEP_DEPS) $(MATH_LIBS) -o $@

# --- LIBCUBA Benchmark ---
ifndef CONDA_PREFIX
$(error CONDA_PREFIX is not set!)
endif

CUBA_CXXFLAGS = -Wall -O2 -std=c++11
CUBA_INCLUDES = -I$(CONDA_PREFIX)/include
CUBA_LIBDIRS = -L$(CONDA_PREFIX)/lib
CUBA_LIBS = -lcuba -lm

cuba_example: cuba_example.cpp
	@echo "Building libcuba example..."
	$(CXX) $(CUBA_CXXFLAGS) $(CUBA_INCLUDES) $(CUBA_LIBDIRS) $< $(CUBA_LIBS) -o $@

run_cuba: cuba_example
	@echo "Running cuba_example..."
	@export LD_LIBRARY_PATH=$(CONDA_PREFIX)/lib:$$LD_LIBRARY_PATH && ./cuba_example

.PHONY: all clean run_cuba

clean:
	@echo "Cleaning benchmarks..."
	rm -f $(TARGETS) *.o
