use criterion::{black_box, criterion_group, criterion_main, Criterion};
use wide::f64x4;

use mchep::integrand::{Integrand, SimdIntegrand};
use mchep::vegas::Vegas;

// Scalar integrand
struct GaussianIntegrand;

impl Integrand for GaussianIntegrand {
    fn dim(&self) -> usize {
        2
    }
    fn eval(&self, x: &[f64]) -> f64 {
        (-(x[0].powi(2)) - x[1].powi(2)).exp()
    }
}

// SIMD integrand
struct GaussianSimdIntegrand;
impl SimdIntegrand for GaussianSimdIntegrand {
    fn dim(&self) -> usize {
        2
    }
    fn eval_simd(&self, points: &[f64x4]) -> f64x4 {
        let x = points[0];
        let y = points[1];
        (-(x * x) - (y * y)).exp()
    }
}

// Scalar 3D integrand
struct Complex3DIntegrand;
impl Integrand for Complex3DIntegrand {
    fn dim(&self) -> usize {
        3
    }
    fn eval(&self, x: &[f64]) -> f64 {
        x[0].sin() * x[1].cos() * (-(x[2] * x[2])).exp()
    }
}

// SIMD 3D integrand
struct Complex3DSimdIntegrand;
impl SimdIntegrand for Complex3DSimdIntegrand {
    fn dim(&self) -> usize {
        3
    }
    fn eval_simd(&self, points: &[f64x4]) -> f64x4 {
        let x = points[0];
        let y = points[1];
        let z = points[2];
        x.sin() * y.cos() * (-(z * z)).exp()
    }
}

fn vegas_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Vegas 2D Gaussian");
    let boundaries2d = &[(-1.0, 1.0), (-1.0, 1.0)];
    let n_eval2d = 100_000;

    group.bench_function("Scalar (Rayon)", |b| {
        b.iter(|| {
            let mut vegas = Vegas::new(1, n_eval2d, 50, 0.5, boundaries2d);
            vegas.set_seed(1234);
            vegas.integrate(black_box(&GaussianIntegrand));
        })
    });

    group.bench_function("SIMD", |b| {
        b.iter(|| {
            let mut vegas = Vegas::new(1, n_eval2d, 50, 0.5, boundaries2d);
            vegas.set_seed(1234);
            vegas.integrate_simd(black_box(&GaussianSimdIntegrand));
        })
    });
    group.finish();

    let mut group2 = c.benchmark_group("Vegas 3D Complex");
    let boundaries3d = &[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)];
    let n_eval3d = 100_000;

    group2.bench_function("Scalar (Rayon)", |b| {
        b.iter(|| {
            let mut vegas = Vegas::new(1, n_eval3d, 50, 0.5, boundaries3d);
            vegas.set_seed(1234);
            vegas.integrate(black_box(&Complex3DIntegrand));
        })
    });

    group2.bench_function("SIMD", |b| {
        b.iter(|| {
            let mut vegas = Vegas::new(1, n_eval3d, 50, 0.5, boundaries3d);
            vegas.set_seed(1234);
            vegas.integrate_simd(black_box(&Complex3DSimdIntegrand));
        })
    });
    group2.finish();
}

criterion_group!(benches, vegas_benchmark);
criterion_main!(benches);
