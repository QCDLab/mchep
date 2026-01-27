CXX = c++ -std=c++11
MPICXX = mpicxx -std=c++14
CXXFLAGS = -O3 -Wall -Wextra
MCHEP_DEPS = $(shell pkg-config --cflags --libs mchep_capi)
MATH_LIBS = -lm

MCHEP_TARGETS = benchmark_mchep benchmark_mchep_simd benchmark_mchep_accuracy benchmark_mchep_simd_accuracy benchmark_scaling benchmark_simd_avx benchmark_heavy_qcd
MPI_TARGETS = benchmark_mpi_simd_avx
CUBA_TARGETS = cuba_example cuba_accuracy cuba_scaling benchmark_heavy_cuba
TARGETS = $(MCHEP_TARGETS) $(CUBA_TARGETS)

all: $(TARGETS)

# --- MCHEP Benchmarks ---
benchmark_mchep: benchmark_mchep.cpp
	$(CXX) $(CXXFLAGS) $< $(MCHEP_DEPS) $(MATH_LIBS) -o $@

benchmark_mchep_simd: benchmark_mchep_simd.cpp
	$(CXX) $(CXXFLAGS) $< $(MCHEP_DEPS) $(MATH_LIBS) -o $@

benchmark_mchep_accuracy: benchmark_mchep_accuracy.cpp
	$(CXX) $(CXXFLAGS) $< $(MCHEP_DEPS) $(MATH_LIBS) -o $@

benchmark_mchep_simd_accuracy: benchmark_mchep_simd_accuracy.cpp
	$(CXX) $(CXXFLAGS) $< $(MCHEP_DEPS) $(MATH_LIBS) -o $@

benchmark_scaling: benchmark_scaling.cpp
	$(CXX) $(CXXFLAGS) $< $(MCHEP_DEPS) $(MATH_LIBS) -o $@

benchmark_simd_avx: benchmark_simd_avx.cpp
	c++ -std=c++14 -O3 -march=native -mavx2 -ffast-math $< $(MCHEP_DEPS) $(MATH_LIBS) -o $@

benchmark_heavy_qcd: benchmark_heavy_qcd.cpp
	c++ -std=c++14 -O3 -march=native -mavx2 -ffast-math $< $(MCHEP_DEPS) $(MATH_LIBS) -o $@

# --- MPI Benchmarks ---
# Note: MPI targets are not included in 'all' by default.
# Build with: make mpi
# Requires mchep_capi to be built with MPI support:
#   cargo build -p mchep_capi --features mpi --release
mpi: $(MPI_TARGETS)

benchmark_mpi_simd_avx: benchmark_mpi_simd_avx.cpp
	$(MPICXX) -O3 -march=native -mavx2 -ffast-math -DMCHEP_MPI $< $(MCHEP_DEPS) $(MATH_LIBS) -o $@

# --- LIBCUBA Benchmark ---
ifndef CONDA_PREFIX
$(error CONDA_PREFIX is not set!)
endif

CUBA_CXXFLAGS = -Wall -O2 -std=c++11
CUBA_INCLUDES = -I$(CONDA_PREFIX)/include
CUBA_LIBDIRS = -L$(CONDA_PREFIX)/lib
CUBA_RPATH = -Wl,-rpath,$(CONDA_PREFIX)/lib
CUBA_LIBS = -lcuba -lm

cuba_example: cuba_example.cpp
	@echo "Building libcuba example..."
	$(CXX) $(CUBA_CXXFLAGS) $(CUBA_INCLUDES) $(CUBA_LIBDIRS) $(CUBA_RPATH) $< $(CUBA_LIBS) -o $@

cuba_accuracy: cuba_accuracy.cpp
	@echo "Building cuba accuracy benchmark..."
	$(CXX) $(CUBA_CXXFLAGS) $(CUBA_INCLUDES) $(CUBA_LIBDIRS) $(CUBA_RPATH) $< $(CUBA_LIBS) -o $@

cuba_scaling: cuba_scaling.cpp
	@echo "Building cuba scaling benchmark..."
	$(CXX) $(CUBA_CXXFLAGS) $(CUBA_INCLUDES) $(CUBA_LIBDIRS) $(CUBA_RPATH) $< $(CUBA_LIBS) -o $@

benchmark_heavy_cuba: benchmark_heavy_cuba.cpp
	@echo "Building heavy CUBA benchmark..."
	c++ -std=c++14 -O3 -march=native -ffast-math $(CUBA_INCLUDES) $(CUBA_LIBDIRS) $(CUBA_RPATH) $< $(CUBA_LIBS) -o $@

run_cuba: cuba_example
	@echo "Running cuba_example..."
	@./cuba_example

.PHONY: all mpi clean run_cuba

clean:
	@echo "Cleaning benchmarks..."
	rm -f $(TARGETS) $(MPI_TARGETS) *.o
