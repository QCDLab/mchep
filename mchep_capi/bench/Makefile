# MCHEP Benchmark Makefile
# Works on Linux and macOS (x86_64 and ARM64)

# ============================================================================
# Detect OS and Architecture
# ============================================================================
UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)

# ============================================================================
# Compiler Configuration
# ============================================================================
CXX ?= c++
MPICXX ?= mpicxx

CXXFLAGS_BASE = -O3 -Wall -Wextra
CXXFLAGS_CXX11 = -std=c++11 $(CXXFLAGS_BASE)
CXXFLAGS_CXX14 = -std=c++14 $(CXXFLAGS_BASE)

# Architecture-specific flags
ifeq ($(UNAME_M),x86_64)
    # x86_64: Enable AVX2 and FMA
    SIMD_FLAGS = -march=native -mavx2 -mfma -ffast-math
else ifeq ($(UNAME_M),aarch64)
    # Linux ARM64: Enable NEON (usually on by default)
    SIMD_FLAGS = -march=native -ffast-math
else ifeq ($(UNAME_M),arm64)
    # macOS ARM64 (Apple Silicon): Enable NEON
    SIMD_FLAGS = -march=native -ffast-math
else
    # Fallback: just native optimizations
    SIMD_FLAGS = -march=native -ffast-math
endif

# ============================================================================
# MCHEP Dependencies (via pkg-config)
# ============================================================================
MCHEP_DEPS = $(shell pkg-config --cflags --libs mchep_capi 2>/dev/null)
ifeq ($(MCHEP_DEPS),)
    $(warning pkg-config could not find mchep_capi. Make sure it is installed and PKG_CONFIG_PATH is set.)
endif

MATH_LIBS = -lm

# ============================================================================
# Targets
# ============================================================================
MCHEP_TARGETS = benchmark_mchep benchmark_mchep_simd benchmark_mchep_accuracy \
                benchmark_mchep_simd_accuracy benchmark_scaling

# AVX targets only on x86_64
ifeq ($(UNAME_M),x86_64)
    MCHEP_TARGETS += benchmark_simd_avx benchmark_heavy_qcd
    MPI_TARGETS = benchmark_mpi_simd_avx
else
    MPI_TARGETS =
endif

CUBA_TARGETS = cuba_example cuba_accuracy cuba_scaling benchmark_heavy_cuba

# Default target: only MCHEP (not CUBA, which requires CONDA_PREFIX)
all: $(MCHEP_TARGETS)

# ============================================================================
# MCHEP Benchmarks
# ============================================================================
benchmark_mchep: benchmark_mchep.cpp
	$(CXX) $(CXXFLAGS_CXX11) $< $(MCHEP_DEPS) $(MATH_LIBS) -o $@

benchmark_mchep_simd: benchmark_mchep_simd.cpp
	$(CXX) $(CXXFLAGS_CXX11) $< $(MCHEP_DEPS) $(MATH_LIBS) -o $@

benchmark_mchep_accuracy: benchmark_mchep_accuracy.cpp
	$(CXX) $(CXXFLAGS_CXX11) $< $(MCHEP_DEPS) $(MATH_LIBS) -o $@

benchmark_mchep_simd_accuracy: benchmark_mchep_simd_accuracy.cpp
	$(CXX) $(CXXFLAGS_CXX11) $< $(MCHEP_DEPS) $(MATH_LIBS) -o $@

benchmark_scaling: benchmark_scaling.cpp
	$(CXX) $(CXXFLAGS_CXX11) $< $(MCHEP_DEPS) $(MATH_LIBS) -o $@

# AVX benchmarks (x86_64 only)
benchmark_simd_avx: benchmark_simd_avx.cpp
	$(CXX) $(CXXFLAGS_CXX14) $(SIMD_FLAGS) $< $(MCHEP_DEPS) $(MATH_LIBS) -o $@

benchmark_heavy_qcd: benchmark_heavy_qcd.cpp
	$(CXX) $(CXXFLAGS_CXX14) $(SIMD_FLAGS) $< $(MCHEP_DEPS) $(MATH_LIBS) -o $@

# ============================================================================
# MPI Benchmarks
# ============================================================================
# Not included in 'all' by default.
# Build with: make mpi
# Requires: cargo build -p mchep_capi --features mpi --release

.PHONY: mpi
mpi: $(MPI_TARGETS)
ifeq ($(MPI_TARGETS),)
	@echo "MPI+AVX benchmarks only available on x86_64 architecture"
endif

benchmark_mpi_simd_avx: benchmark_mpi_simd_avx.cpp
	$(MPICXX) $(CXXFLAGS_CXX14) $(SIMD_FLAGS) -DMCHEP_MPI $< $(MCHEP_DEPS) $(MATH_LIBS) -o $@

# ============================================================================
# CUBA Benchmarks (optional, requires CONDA_PREFIX)
# ============================================================================
.PHONY: cuba
cuba: check-conda $(CUBA_TARGETS)

.PHONY: check-conda
check-conda:
ifndef CONDA_PREFIX
	$(error CONDA_PREFIX is not set! CUBA benchmarks require libcuba from conda.)
endif

CUBA_CXXFLAGS = -Wall -O2 -std=c++11
CUBA_INCLUDES = -I$(CONDA_PREFIX)/include
CUBA_LIBDIRS = -L$(CONDA_PREFIX)/lib

# OS-specific rpath flag
ifeq ($(UNAME_S),Darwin)
    CUBA_RPATH = -Wl,-rpath,$(CONDA_PREFIX)/lib
else
    CUBA_RPATH = -Wl,-rpath,$(CONDA_PREFIX)/lib -Wl,--enable-new-dtags
endif

CUBA_LIBS = -lcuba -lm

cuba_example: cuba_example.cpp | check-conda
	@echo "Building libcuba example..."
	$(CXX) $(CUBA_CXXFLAGS) $(CUBA_INCLUDES) $(CUBA_LIBDIRS) $(CUBA_RPATH) $< $(CUBA_LIBS) -o $@

cuba_accuracy: cuba_accuracy.cpp | check-conda
	@echo "Building cuba accuracy benchmark..."
	$(CXX) $(CUBA_CXXFLAGS) $(CUBA_INCLUDES) $(CUBA_LIBDIRS) $(CUBA_RPATH) $< $(CUBA_LIBS) -o $@

cuba_scaling: cuba_scaling.cpp | check-conda
	@echo "Building cuba scaling benchmark..."
	$(CXX) $(CUBA_CXXFLAGS) $(CUBA_INCLUDES) $(CUBA_LIBDIRS) $(CUBA_RPATH) $< $(CUBA_LIBS) -o $@

benchmark_heavy_cuba: benchmark_heavy_cuba.cpp | check-conda
	@echo "Building heavy CUBA benchmark..."
	$(CXX) $(CXXFLAGS_CXX14) $(SIMD_FLAGS) $(CUBA_INCLUDES) $(CUBA_LIBDIRS) $(CUBA_RPATH) $< $(CUBA_LIBS) -o $@

# ============================================================================
# Utility Targets
# ============================================================================
.PHONY: run_cuba
run_cuba: cuba_example
	@echo "Running cuba_example..."
	@./cuba_example

.PHONY: info
info:
	@echo "OS:           $(UNAME_S)"
	@echo "Architecture: $(UNAME_M)"
	@echo "CXX:          $(CXX)"
	@echo "MPICXX:       $(MPICXX)"
	@echo "SIMD_FLAGS:   $(SIMD_FLAGS)"
	@echo "MCHEP_DEPS:   $(MCHEP_DEPS)"
	@echo ""
	@echo "MCHEP targets: $(MCHEP_TARGETS)"
	@echo "MPI targets:   $(MPI_TARGETS)"
	@echo "CUBA targets:  $(CUBA_TARGETS) (requires 'make cuba' and CONDA_PREFIX)"

.PHONY: all mpi cuba clean info

clean:
	@echo "Cleaning benchmarks..."
	rm -f $(MCHEP_TARGETS) $(MPI_TARGETS) $(CUBA_TARGETS) *.o
