#ifndef MCHEP_HPP
#define MCHEP_HPP

#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <utility>
#include <vector>

extern "C" {
#include <mchep_capi.h>
}

namespace mchep {

// Forward declaration
class Vegas;

namespace internal {
// This wrapper function is the bridge between C-style function pointers and
// std::function. It will be passed to the C API.
extern "C" double integrand_wrapper(const double *x, int dim, void *user_data) {
  if (user_data == nullptr) {
    // Or handle error appropriately
    return 0.0;
  }
  auto *func =
      static_cast<std::function<double(const std::vector<double> &)> *>(
          user_data);
  std::vector<double> x_vec(x, x + dim);
  return (*func)(x_vec);
}

// SIMD wrapper
extern "C" void integrand_simd_wrapper(const double *x, int dim,
                                       void *user_data, double *result) {
  if (user_data == nullptr) {
    return;
  }
  auto *func = static_cast<std::function<
      std::array<double, 4>(const std::vector<double> &)> *>(user_data);
  std::vector<double> x_vec(x, x + dim * 4);
  std::array<double, 4> res_arr = (*func)(x_vec);
  std::copy(res_arr.begin(), res_arr.end(), result);
}
} // namespace internal

class Vegas {
public:
  /// @brief Constructor for the Vegas integrator.
  /// @param n_iter Number of iterations.
  /// @param n_eval Number of function evaluations per iteration.
  /// @param n_bins Number of bins for the adaptive grid.
  /// @param alpha The grid adaptation parameter.
  /// @param boundaries The integration boundaries for each dimension.
  Vegas(size_t n_iter, size_t n_eval, size_t n_bins, double alpha,
        const std::vector<std::pair<double, double>> &boundaries) {
    std::vector<CBoundary> c_boundaries;
    c_boundaries.reserve(boundaries.size());
    for (const auto &b : boundaries) {
      c_boundaries.push_back({b.first, b.second});
    }

    vegas_ptr_ = mchep_vegas_new(n_iter, n_eval, n_bins, alpha,
                               c_boundaries.size(), c_boundaries.data());
    if (vegas_ptr_ == nullptr) {
      throw std::runtime_error("Failed to create MCHEP Vegas integrator.");
    }
  }

  /// @brief Destructor.
  ~Vegas() { mchep_vegas_free(vegas_ptr_); }

  // Delete copy constructor and copy assignment operator
  Vegas(const Vegas &) = delete;
  Vegas &operator=(const Vegas &) = delete;

  /// @brief Move constructor.
  Vegas(Vegas &&other) noexcept : vegas_ptr_(other.vegas_ptr_) {
    other.vegas_ptr_ = nullptr;
  }

  /// @brief Move assignment operator.
  Vegas &operator=(Vegas &&other) noexcept {
    if (this != &other) {
      mchep_vegas_free(vegas_ptr_);
      vegas_ptr_ = other.vegas_ptr_;
      other.vegas_ptr_ = nullptr;
    }
    return *this;
  }

  /// @brief Sets the seed for the random number generator.
  /// @param seed The seed to use.
  void set_seed(uint64_t seed) { mchep_vegas_set_seed(vegas_ptr_, seed); }

  /// @brief Integrates the given function.
  /// @param integrand The function to integrate. It should take a vector of
  /// doubles and return a double.
  /// @return The integration result.
  VegasResult
  integrate(std::function<double(const std::vector<double> &)> integrand,
            double target_accuracy = -1.0) {
    return mchep_vegas_integrate(vegas_ptr_, internal::integrand_wrapper,
                                 &integrand, target_accuracy);
  }

  /// @brief Integrates the given function using SIMD.
  /// @param integrand The function to integrate. It should take a vector of
  /// doubles of size dim*4 (SoA) and return an array of 4 doubles.
  /// @param target_accuracy The desired accuracy in percent. If non-positive, it is ignored.
  /// @return The integration result.
  VegasResult integrate_simd(
      std::function<std::array<double, 4>(const std::vector<double> &)>
          integrand,
      double target_accuracy = -1.0) {
    return mchep_vegas_integrate_simd(vegas_ptr_,
                                      internal::integrand_simd_wrapper,
                                      &integrand, target_accuracy);
  }

private:
  VegasC *vegas_ptr_;
};

} // namespace mchep

#endif // MCHEP_HPP
