//! Standard benchmark integrands for Monte Carlo integration.
use crate::integrand::{BurnIntegrand, Integrand, SimdIntegrand};
use burn::prelude::{Backend, Tensor};
use wide::f64x4;

use libm::erf;
use std::f64::consts::PI;

/// Returns the parameters for the Genz test functions.
fn genz_params(dim: usize) -> (Vec<f64>, Vec<f64>) {
    let mut c = Vec::with_capacity(dim);
    let mut w = Vec::with_capacity(dim);
    for i in 1..=dim {
        c.push(((i as f64).sqrt() * 12345.).fract());
        w.push(((i as f64).sqrt() * 54321.).fract());
    }
    (c, w)
}

// Genz Oscillatory
#[derive(Clone)]
pub struct GenzOscillatory {
    dim: usize,
    c: Vec<f64>,
    w: Vec<f64>,
    c_simd: Vec<f64x4>,
    w_simd: Vec<f64x4>,
}

impl GenzOscillatory {
    pub fn new(dim: usize) -> Self {
        let (c, w) = genz_params(dim);
        let c_simd = c.iter().map(|&val| f64x4::splat(val)).collect();
        let w_simd = w.iter().map(|&val| f64x4::splat(val)).collect();
        Self {
            dim,
            c,
            w,
            c_simd,
            w_simd,
        }
    }

    pub fn analytical_result(&self) -> f64 {
        let mut result = 1.0;
        for i in 0..self.dim {
            result *= (self.c[i] * (1. - self.w[i])).sin() + (self.c[i] * self.w[i]).sin();
            result /= self.c[i];
        }
        result
    }
}

impl Integrand for GenzOscillatory {
    fn dim(&self) -> usize {
        self.dim
    }

    fn eval(&self, x: &[f64]) -> f64 {
        let mut sum = 0.;
        for i in 0..self.dim {
            sum += self.c[i] * (x[i] - self.w[i]);
        }
        sum.cos()
    }
}

impl SimdIntegrand for GenzOscillatory {
    fn dim(&self) -> usize {
        self.dim
    }

    fn eval_simd(&self, x: &[f64x4]) -> f64x4 {
        let mut sum = f64x4::splat(0.);
        for i in 0..self.dim {
            sum += self.c_simd[i] * (x[i] - self.w_simd[i]);
        }
        sum.cos()
    }
}

#[cfg(feature = "gpu")]
impl<B: Backend> BurnIntegrand<B> for GenzOscillatory {
    fn dim(&self) -> usize {
        self.dim
    }

    fn eval_burn(&self, x: Tensor<B, 2>) -> Tensor<B, 1> {
        let c = Tensor::<B, 1>::from_floats(
            self.c.iter().map(|&val| val as f32).collect::<Vec<_>>().as_slice(),
            &x.device(),
        )
        .reshape([1, self.dim]);
        let w = Tensor::<B, 1>::from_floats(
            self.w.iter().map(|&val| val as f32).collect::<Vec<_>>().as_slice(),
            &x.device(),
        )
        .reshape([1, self.dim]);

        let c = c.expand(x.dims());
        let w = w.expand(x.dims());

        let sum = c.mul(x.sub(w)).sum_dim(1);
        sum.cos().squeeze()
    }
}

// Genz Product Peak
#[derive(Clone)]
pub struct GenzProductPeak {
    dim: usize,
    c: Vec<f64>,
    w: Vec<f64>,
    c_simd: Vec<f64x4>,
    w_simd: Vec<f64x4>,
}

impl GenzProductPeak {
    pub fn new(dim: usize) -> Self {
        let (c, w) = genz_params(dim);
        let c_simd = c.iter().map(|&val| f64x4::splat(val)).collect();
        let w_simd = w.iter().map(|&val| f64x4::splat(val)).collect();
        Self {
            dim,
            c,
            w,
            c_simd,
            w_simd,
        }
    }

    pub fn analytical_result(&self) -> f64 {
        let mut result = 1.0;
        for i in 0..self.dim {
            result *= self.c[i]
                * ((1. - self.w[i]).atan() * self.c[i] + (self.w[i]).atan() * self.c[i]);
        }
        result
    }
}

impl Integrand for GenzProductPeak {
    fn dim(&self) -> usize {
        self.dim
    }

    fn eval(&self, x: &[f64]) -> f64 {
        let mut result = 1.0;
        for i in 0..self.dim {
            result *= 1.0 / (self.c[i].powi(-2) + (x[i] - self.w[i]).powi(2));
        }
        result
    }
}

impl SimdIntegrand for GenzProductPeak {
    fn dim(&self) -> usize {
        self.dim
    }

    fn eval_simd(&self, x: &[f64x4]) -> f64x4 {
        let mut result = f64x4::splat(1.0);
        for i in 0..self.dim {
            result *= f64x4::splat(1.0) / (self.c_simd[i].powf(-2.) + (x[i] - self.w_simd[i]).powf(2.));
        }
        result
    }
}

#[cfg(feature = "gpu")]
impl<B: Backend> BurnIntegrand<B> for GenzProductPeak {
    fn dim(&self) -> usize {
        self.dim
    }

    fn eval_burn(&self, x: Tensor<B, 2>) -> Tensor<B, 1> {
        let c = Tensor::<B, 1>::from_floats(
            self.c.iter().map(|&val| val as f32).collect::<Vec<_>>().as_slice(),
            &x.device(),
        )
        .reshape([1, self.dim]);
        let w = Tensor::<B, 1>::from_floats(
            self.w.iter().map(|&val| val as f32).collect::<Vec<_>>().as_slice(),
            &x.device(),
        )
        .reshape([1, self.dim]);

        let c = c.expand(x.dims());
        let w = w.expand(x.dims());

        let result = c
            .powf_scalar(-2.0)
            .add(x.sub(w).powf_scalar(2.0))
            .recip()
            .prod_dim(1);
        result.squeeze()
    }
}

// Genz Gaussian
#[derive(Clone)]
pub struct GenzGaussian {
    dim: usize,
    c: Vec<f64>,
    w: Vec<f64>,
    c_simd: Vec<f64x4>,
    w_simd: Vec<f64x4>,
}

impl GenzGaussian {
    pub fn new(dim: usize) -> Self {
        let (c, w) = genz_params(dim);
        let c_simd = c.iter().map(|&val| f64x4::splat(val)).collect();
        let w_simd = w.iter().map(|&val| f64x4::splat(val)).collect();
        Self {
            dim,
            c,
            w,
            c_simd,
            w_simd,
        }
    }

    pub fn analytical_result(&self) -> f64 {
        let mut result = 1.0;
        for i in 0..self.dim {
            result *= (PI.sqrt() / (2. * self.c[i]))
                * (erf(2. * self.c[i] * (1. - self.w[i])) + erf(2. * self.c[i] * self.w[i]));
        }
        result
    }
}

impl Integrand for GenzGaussian {
    fn dim(&self) -> usize {
        self.dim
    }

    fn eval(&self, x: &[f64]) -> f64 {
        let mut sum = 0.0;
        for i in 0..self.dim {
            sum += self.c[i].powi(2) * (x[i] - self.w[i]).powi(2);
        }
        (-sum).exp()
    }
}

impl SimdIntegrand for GenzGaussian {
    fn dim(&self) -> usize {
        self.dim
    }

    fn eval_simd(&self, x: &[f64x4]) -> f64x4 {
        let mut sum = f64x4::splat(0.0);
        for i in 0..self.dim {
            sum += self.c_simd[i].powf(2.) * (x[i] - self.w_simd[i]).powf(2.);
        }
        (-sum).exp()
    }
}

#[cfg(feature = "gpu")]
impl<B: Backend> BurnIntegrand<B> for GenzGaussian {
    fn dim(&self) -> usize {
        self.dim
    }

    fn eval_burn(&self, x: Tensor<B, 2>) -> Tensor<B, 1> {
        let c = Tensor::<B, 1>::from_floats(
            self.c.iter().map(|&val| val as f32).collect::<Vec<_>>().as_slice(),
            &x.device(),
        )
        .reshape([1, self.dim]);
        let w = Tensor::<B, 1>::from_floats(
            self.w.iter().map(|&val| val as f32).collect::<Vec<_>>().as_slice(),
            &x.device(),
        )
        .reshape([1, self.dim]);

        let c = c.expand(x.dims());
        let w = w.expand(x.dims());

        let sum = c.powf_scalar(2.0).mul(x.sub(w).powf_scalar(2.0)).sum_dim(1);
        sum.mul_scalar(-1.0).exp().squeeze()
    }
}

// Endpoint Singularity
#[derive(Clone)]
pub struct EndpointSingularity {
    dim: usize,
}

impl EndpointSingularity {
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }

    pub fn analytical_result(&self) -> f64 {
        2.0f64.powi(self.dim as i32)
    }
}

impl Integrand for EndpointSingularity {
    fn dim(&self) -> usize {
        self.dim
    }

    fn eval(&self, x: &[f64]) -> f64 {
        let mut result = 1.0;
        for &val in x {
            if val <= 0.0 {
                return 0.0;
            }
            result *= val.powf(-0.5);
        }
        result
    }
}

impl SimdIntegrand for EndpointSingularity {
    fn dim(&self) -> usize {
        self.dim
    }

    fn eval_simd(&self, x: &[f64x4]) -> f64x4 {
        let mut result = f64x4::splat(1.0);
        for &val in x {
            result *= val.powf(-0.5);
        }
        result
    }
}

#[cfg(feature = "gpu")]
impl<B: Backend> BurnIntegrand<B> for EndpointSingularity {
    fn dim(&self) -> usize {
        self.dim
    }

    fn eval_burn(&self, x: Tensor<B, 2>) -> Tensor<B, 1> {
        x.powf_scalar(-0.5).prod_dim(1).squeeze()
    }
}
