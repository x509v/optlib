#include <gtest/gtest.h>
#include <optlib/methods/genetic.hpp>

namespace optlib::test {

// TODO: add tests once genetic() is implemented
//
// TEST(Genetic, Sphere) {
//     // f(x) = ||x||^2, bounded search in [-5, 5]^n
//     Vector lo = -5.0 * Vector::Ones(5);
//     Vector hi =  5.0 * Vector::Ones(5);
//     auto r = genetic([](const Vector& x){ return x.squaredNorm(); }, lo, hi);
//     EXPECT_LT(r.f_val, 1e-4);
// }

} // namespace optlib::test
