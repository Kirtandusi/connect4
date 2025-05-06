use crate::connect4_env::Connect4Env;
use crate::game_state::GameState;
use crate::player::Player;
use rand::Rng;
pub(crate) struct Neuron {
    weights: Vec<f64>,          // Weights for each input
    bias: f64,                  // Bias value
    activation: fn(f64) -> f64, // Activation function
    output: f64,                //activated output
    z: f64,                     //raw input before activation. needed for back propagation.
}
impl Neuron {
    pub fn new(weights: Vec<f64>, bias: f64, activation: fn(f64) -> f64) -> Self {
        Self {
            weights,
            bias,
            activation,
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
}
type Layer = Vec<Neuron>; //each layer is a vector of Neurons
pub struct NeuralNetwork {
    layers: Vec<Layer>,
    learning_rate: f64,
}

impl NeuralNetwork {
    pub fn new(layers: Vec<Layer>, learning_rate: f64) -> Self {
        Self {
            layers,
            learning_rate,
        }
    }
    pub fn train(&mut self, env: &mut Connect4Env) {
        let mut epsilon = 1.0; //start epsilon high to encourage exploration
        let epsilon_min = 0.01;
        let epsilon_decay = 0.999;
        let episodes = 30000;
        let learning_rate = 0.1;
        let discount_factor = 0.99;

        //iterate through episodes
        for _ in 0..episodes {
            env.reset();
            let mut done = false;
            let mut state = env.get_state_vector(); //get initial state

            //while not terminal state
            while !done {
                let action = if rand::random::<f64>() < epsilon { //epsilon greedy.
                    //if < epsilon, choose random action. This encourages exploration.
                    env.sample_random_action()
                } else {
                    let q_values = self.forward(&state); //use dqn when >= epsilon
                    self.argmax_valid_action(&q_values, &env)
                };

                let (next_state, reward, is_done) = env.step(action);
                done = is_done;

                let next_q_values = self.forward(&next_state);
                let max_next_q = next_q_values
                    .iter()
                    .cloned()
                    .fold(f64::NEG_INFINITY, f64::max);
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
        target
            .iter()
            .zip(prediction.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f64>()
            / target.len() as f64
    }
    /*
    forward pass. Calculates z = w * x + b. applies activation, stores output.
     */
    pub fn forward(&mut self, input: &Vec<f64>) -> Vec<f64> {
        let mut current_input = input.clone();

        for layer in &mut self.layers {
            let mut next_input = vec![];
            for neuron in layer {
                let z = neuron
                    .weights
                    .iter()
                    .zip(&current_input)
                    .map(|(w, i)| w * i)
                    .sum::<f64>()
                    + neuron.bias;
                neuron.z = z;
                neuron.output = (neuron.activation)(z);
                next_input.push(neuron.output);
            }
            current_input = next_input;
        }

        current_input
    }
    pub fn back(&mut self, input: &Vec<f64>, target: &Vec<f64>) {}

    fn argmax_valid_action(&self, q_values: &Vec<f64>, env: &Connect4Env) -> usize {
        let valid = env.valid_moves();
        *valid
            .iter()
            .max_by(|&&a, &&b| q_values[a].partial_cmp(&q_values[b]).unwrap())
            .unwrap()
    }
}
pub struct NeuralNetPlayer {
    player: bool,
    pub network: NeuralNetwork,
}

//deep Q learning implementation. Values are set to be random instead of 0. Will resolve naturally.
// learning rate implemented during back prop, not needed during Q learning.
impl NeuralNetPlayer {
    pub fn new(player: bool) -> Self {
        let input_size = 42; //connect 4 is 7x6
        let hidden_size = 10; //just a random number - CHANGE THIS MAYBE?
        let output_size = 7; //7 columns AI can move.
        let mut rng = rand::rng();

        //initialize hidden layer.
        //random weights, biases
        let hidden_layer: Layer = (0..hidden_size)
            .map(|_| {
                let weights: Vec<f64> = (0..input_size)
                    .map(|_| rng.random_range(-1.0..1.0))
                    .collect();
                let bias = rng.random_range(-1.0..1.0);
                Neuron::new(weights, bias, Neuron::relu_activation)
            })
            .collect();
        let output_layer: Layer = (0..output_size)
            .map(|_| {
                let weights: Vec<f64> = (0..hidden_size)
                    .map(|_| rng.random_range(-1.0..1.0))
                    .collect();
                let bias = rng.random_range(-1.0..1.0);
                Neuron::new(weights, bias, Neuron::relu_activation)
            })
            .collect();

        let network = NeuralNetwork {
            layers: vec![hidden_layer, output_layer],
            learning_rate: 0.1,
        };

        Self { player, network }
    }
}
//actual player impl.
impl Player for NeuralNetPlayer {
    fn make_move(&mut self, game_state: &mut GameState) {
        let input = game_state.to_input_vector();
        let q_values = self.network.forward(&input);

        let valid_moves = game_state.get_valid_moves();
        let best_action = *valid_moves
            .iter()
            .max_by(|&&a, &&b| q_values[a].partial_cmp(&q_values[b]).unwrap())
            .unwrap();

        game_state.play_move(best_action, self.player);
    }
    fn get_name(&self) -> &str {
        "Neural Net Player"
    }
}
