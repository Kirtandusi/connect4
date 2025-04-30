use crate::game_state::GameState;
use crate::player::Player;
use rand::Rng;
use connect4::connect4_env::Connect4Env;
use connect4_env::Connect4Env;
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
    fn train(&mut self, env: &mut Connect4Env) {
        let mut epsilon = 1.0;
        let epsilon_min = 0.01;
        let epsilon_decay = 0.999;
        let episodes = 30000;
        let learning_rate = 0.1;
        let discount_factor = 0.99;

        for i in 0..episodes {
            env.reset();
            let mut done = false;
            let mut state = env.get_state_vector();

            while !done {
                let action = if rand::random::<f64>() < epsilon {
                    env.sample_random_action()
                } else {
                    let q_values = self.forward(&state);
                    self.argmax_valid_action(&q_values, &env)
                };

                let (next_state, reward, is_done) = env.step(action);
                done = is_done;

                let next_q_values = self.forward(&next_state);
                let max_next_q = next_q_values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let mut target_q_values = self.forward(&state);
                target_q_values[action] = reward + discount_factor * max_next_q;

                self.back(&state, &target_q_values);
                state = next_state;
            }
            epsilon = epsilon.min(epsilon * epsilon_decay).max(epsilon_min);
        }
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
    pub fn back(&mut self, _input: &Vec<f64>, _target: &Vec<f64>) {}
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
        //implement code here!!!

        //initialize hidden layer.
        //random weights, biases
        let hidden_layer: Layer = vec![];
        //initialize output layer
        //random weights, biases,
        let output_layer: Layer = vec![];

        let network = NeuralNetwork {
            layers: vec![hidden_layer, output_layer],
        };

        Self {
            player,
            network,
        }
    }
}
//actual player impl.
impl Player for NeuralNetPlayer {
    fn make_move(&mut self, game_state: &mut GameState) {
        // Implement the o make a move using the neural network
        let input = game_state.to_input_vector();
        let output = self.network.forward(&input);

        let mut best_index = 0;
        let mut best_value = output[0];

        for (i, val) in output.iter().enumerate() {
            if val > &best_value {
                best_index = i;
                best_value = *val;
            }
        }
        let best_action = best_index;

        game_state.play_move(best_action, self.player);
    }
    fn get_name(&self) -> &str {
        "Neural Net Player"
    }
}

fn main() {
    let mut env = Connect4Env::new(true);
    let mut ai_player = NeuralNetPlayer::new(true);
    ai_player.network.train(&mut env);

}