use crate::game_state::GameState;
use crate::player::Player;
use rand::Rng;

struct Neuron {
    weights: Vec<f64>,   // Weights for each input
    bias: f64,           // Bias value
    activation: fn(f64) -> f64, // Activation function
    output: f64, //activated output
    z: f64, //raw input before activation. needed for back propagation.
}
impl Neuron {
    pub fn new(weights: Vec<f64>, bias: f64, activation: fn(f64) -> f64) -> Self {
        Self { weights, bias, activation, output: 0.0, z: 0.0 } //just setting all to zero.
    }
    pub fn relu_activation(x: f64) -> f64 {
        if x > 0.0 { x } else { 0.0 }
    }
}
type Layer = Vec<Neuron>; //each layer is a vector of Neurons
struct NeuralNetwork {
    layers: Vec<Layer>,
}

impl NeuralNetwork {
    pub fn new(layers: Vec<Layer>, learning_rate: f64) -> Self {
        Self { layers }
    }
    fn train() {

    }
    /*
    calculates mean squared error between target and prediction.
     */
    fn mse_loss(target: &Vec<f64>, prediction: &Vec<f64>) -> f64 {
        target.iter().zip(prediction.iter()).map(|(x, y)|
            (x - y).powi(2)).sum::<f64>() / target.len() as f64
    }
    /*
    forward pass. Calculates z = w * x + b. applies activation, stores output.
     */
    pub fn forward(&mut self, input: &Vec<f64>) -> Vec<f64> {
        let mut current_input = input.clone();

        for layer in &mut self.layers {
            let mut next_input = vec![];
            for neuron in layer {
                let z = neuron.weights.iter()
                    .zip(&current_input)
                    .map(|(w, i)| w * i)
                    .sum::<f64>() + neuron.bias;
                neuron.z = z;
                neuron.output = (neuron.activation)(z);
                next_input.push(neuron.output);
            }
            current_input = next_input;
        }

        current_input
    }
    pub fn back(&mut self, _input: &Vec<f64>, _target: &Vec<f64>) {

    }
}
pub struct NeuralNetPlayer {
    player: bool,
    network: NeuralNetwork,
}

//deep Q learning implementation.
impl NeuralNetPlayer {
    pub(crate) fn new(player: bool) -> Self {
        let input_size = 42; //connect 4 is 7x6
        let hidden_size = 10; //just a random number - CHANGE THIS MAYBE?
        let output_size = 7; //7 columns AI can move.
        let mut rng = rand::rng();

        NeuralNetPlayer::new(player)
    }
}
//actual player impl.
impl Player for NeuralNetPlayer {
    fn make_move(&mut self, game_state: &mut GameState) {
        // Implement the o make a move using the neural network
        let input = game_state.to_input_vector();
        let output = self.network.forward(&input);

        let best_action = output.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;

        game_state.play_move(best_action, self.player);
    }
    fn get_name(&self) -> &str {
        "Neural Net Player"
    }
}