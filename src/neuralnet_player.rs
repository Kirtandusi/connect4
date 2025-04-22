use crate::game_state::GameState;
use crate::player::Player;

struct Neuron {
    weights: Vec<f64>,   // Weights for each input
    bias: f64,           // Bias value
    activation: fn(f64) -> f64, // Activation function
    output: f64,
}
impl Neuron {
    pub fn new(weights: Vec<f64>, bias: f64, activation: fn(f64) -> f64) -> Self {
        Self { weights, bias, activation, output: 0.0 }
    }
    pub fn relu_activation(x: f64) -> f64 {
        if x > 0.0 { x } else { 0.0 }
    }
    pub fn sigmoid_activation(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }
}
type Layer = Vec<Neuron>;
struct NeuralNetwork {
    layers: Vec<Layer>,
}
impl NeuralNetwork {
    pub fn new(layers: Vec<Layer>) -> Self {
        Self { layers }
    }
    fn mse_loss() {

    }
    pub fn forward() {

    }
    pub fn back() {

    }
}
pub struct NeuralNetPlayer {
    player: bool,
}

impl NeuralNetPlayer {
    pub(crate) fn new(player: bool) -> Self {
        NeuralNetPlayer { player }
    }
}

impl NeuralNetPlayer {

}
impl Player for NeuralNetPlayer {
    fn make_move(&mut self, _game_state: &mut GameState) {
        // Implement the logic to make a move using the neural network
    }
    fn get_name(&self) -> &str {
        "Neural Net Player"
    }
}