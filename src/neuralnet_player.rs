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
    learning_rate: f64,
}

impl NeuralNetwork {
    pub fn new(layers: Vec<Layer>, learning_rate: f64) -> Self {
        Self { layers, learning_rate }
    }
    fn mse_loss(target: &Vec<f64>, prediction: &Vec<f64>) -> f64 {
        target.iter().zip(prediction.iter()).map(|(x, y)|
            (x - y).powi(2)).sum::<f64>() / target.len() as f64
    }
    pub fn forward(&mut self, input: &Vec<f64>) -> Vec<f64> {
        Vec::new()
    }
    pub fn back(&mut self, _input: &Vec<f64>, _target: &Vec<f64>) {

    }
}
pub struct NeuralNetPlayer {
    player: bool,
}

//deep Q learning
impl NeuralNetPlayer {
    pub(crate) fn new(player: bool) -> Self {
        NeuralNetPlayer { player }
    }
}
impl Player for NeuralNetPlayer {
    fn make_move(&mut self, _game_state: &mut GameState) {
        // Implement the logic to make a move using the neural network
        let input = _game_state.to_input_vector();
        let learning_rate = 0.8;
        let mut network = NeuralNetwork::new(vec![], 0.0);
        let output = network.forward(&input);
        let best_action = output.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        _game_state.play_move(best_action, self.player);
    }
    fn get_name(&self) -> &str {
        "Neural Net Player"
    }
}