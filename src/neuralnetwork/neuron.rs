#[derive(Clone)]
pub struct Neuron {
    pub(crate) weights: Vec<f64>,          // Weights for each input
    pub(crate) bias: f64,                  // Bias value
    pub(crate) activation: fn(f64) -> f64, // Activation function
    pub(crate) activation_derivative: fn(f64) -> f64,
    pub(crate) output: f64,                //activated output
    pub(crate) z: f64,                     //raw input before activation. needed for back propagation.

}
impl Neuron {
    pub fn new(weights: Vec<f64>, bias: f64, activation: fn(f64) -> f64, derivative: fn(f64) -> f64) -> Self {
        Self {
            weights,
            bias,
            activation,
            activation_derivative: derivative,
            output: 0.0,
            z: 0.0,
        } //just setting all to zero, exception activation
    }
    pub fn relu_activation(x: f64) -> f64 {
        if x > 0.0 {
            x
        } else {
            0.0
        }
    }
    pub fn relu_derivative(x: f64) -> f64 {
        if x > 0.0 {
            1.0
        } else {
            0.0
        }
    }
    pub(crate) fn identity(x: f64) -> f64 { x }
    pub(crate) fn identity_derivative(_: f64) -> f64 { 1.0 }

}