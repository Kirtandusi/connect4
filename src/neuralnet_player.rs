use rand::prelude::IndexedRandom;
use crate::game_state::GameState;
use crate::player::Player;
use rand::Rng;
use crate::minimax_player::MinMaxPlayer;
use crate::random_player::RandomPlayer;
use serde::{Serialize, Deserialize}; //to save training data
#[derive(Clone, Serialize, Deserialize)]
pub struct Neuron {
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
    pub fn relu_derivative(x: f64) -> f64 {
        if x > 0.0 {
            1.0
        } else {
            0.0
        }
    }
}
type Layer = Vec<Neuron>; //each layer is a vector of Neurons

#[derive(Clone, Serialize, Deserialize)]
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


    /*
    calculates mean squared error between target and prediction.
     */
    #[allow(dead_code)]
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
    pub fn back(&mut self, input: &Vec<f64>, target: &Vec<f64>) {

        let mut deltas: Vec<Vec<f64>> = Vec::new();

        //first step is to compute deltas for output layer.
        let output_layer = self.layers.last().unwrap();
        let mut output_deltas = vec![];
        //forward called first, every neuron has a z value.
        for (i, neuron) in output_layer.iter().enumerate() {
            let error = neuron.output - target[i];
            let delta = error * Neuron::relu_derivative(neuron.z);
            output_deltas.push(delta);
        }
        deltas.push(output_deltas);

        //then compute deltas for hidden layers (back prop)
        for l in (0..self.layers.len() - 1).rev() {
            let layer = &self.layers[l];
            let next_layer = &self.layers[l + 1];
            let next_deltas = &deltas[0]; // most recent delta is first in list
            let mut layer_deltas = vec![];

            for (i, neuron) in layer.iter().enumerate() {
                // Weighted sum of deltas from next layer
                let mut sum = 0.0;
                for (j, next_neuron) in next_layer.iter().enumerate() {
                    sum += next_neuron.weights[i] * next_deltas[j];
                }
                let delta = sum * Neuron::relu_derivative(neuron.z);
                layer_deltas.push(delta);
            }
            deltas.insert(0, layer_deltas); // prepend
        }

        //update biases, weights
        let mut prev_output = input.clone();
        for (layer_index, layer) in self.layers.iter_mut().enumerate() {
            for (neuron_index, neuron) in layer.iter_mut().enumerate() {
                let delta = deltas[layer_index][neuron_index];
                // Update weights
                for w in 0..neuron.weights.len() {
                    neuron.weights[w] -= self.learning_rate * delta * prev_output[w];
                }
                // Update bias
                neuron.bias -= self.learning_rate * delta;
            }
            // Update prev_output to current layer's output for next layer
            prev_output = layer.iter().map(|n| n.output).collect();
        }
    }

    // fn argmax_valid_action(&self, q_values: &Vec<f64>, state: &GameState) -> usize {
    //     let valid = state.get_valid_moves();
    //     *valid
    //         .iter()
    //         .max_by(|&&a, &&b| q_values[a].partial_cmp(&q_values[b]).unwrap())
    //         .unwrap()
    // }
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

        let network = NeuralNetwork::new(vec![hidden_layer, output_layer], 0.1);

        Self { player, network }
    }
    /*
        using self play, the neural net will train itself.
     */
    pub fn train_generalized(&mut self, episodes: usize) {
        let mut rng = rand::rng();
        let discount = 0.99;
        let mut epsilon = 1.0;
        let epsilon_min = 0.01;
        let epsilon_decay = 0.995;

        for episode in 0..episodes {
            let mut game = GameState::new();
            let mut current_player = true;

            // Choose opponent, 50% random, 30% self-play, 20% Minimax
            let roll: f64 = rng.random(); // between 0.0 and 1.0
            let mut opponent: Box<dyn Player> = if roll < 0.5 {
                Box::new(RandomPlayer::new(!self.player))
            } else if roll < 0.8 {
                // Self-play with a clone of the current network
                let cloned_net = NeuralNetwork {
                    layers: self.network.layers.clone(), // requires Clone on Layer, Neuron
                    learning_rate: self.network.learning_rate,
                };
                Box::new(NeuralNetPlayer {
                    player: !self.player,
                    network: cloned_net,
                })
            } else {
                Box::new(MinMaxPlayer::new(!self.player))
            };

            while game.is_not_full() {
                let state = game.to_input_vector();
                let valid_moves = game.get_valid_moves();

                let action = if current_player == self.player {
                    // Decide move (explore or exploit)
                    if rng.random::<f64>() < epsilon {
                        *valid_moves.choose(&mut rng).unwrap()
                    } else {
                        let q_values = self.network.forward(&state);
                        *valid_moves
                            .iter()
                            .max_by(|&&a, &&b| q_values[a].partial_cmp(&q_values[b]).unwrap())
                            .unwrap()
                    }
                } else {
                    let _prev = game.clone();
                    opponent.make_move(&mut game);
                    let _opp_action = game.get_last_move().unwrap(); // assumes this method exists
                    current_player = !current_player;
                    continue;
                };

                game.play_move(action, current_player);
                let won = game.check_for_win();
                let reward = if won {
                    if current_player == self.player { 1.0 } else { -1.0 }
                } else if !game.is_not_full() {
                    0.5
                } else {
                    0.0
                };

                let next_state = game.to_input_vector();
                let next_q = self.network.forward(&next_state);
                let max_next_q = *next_q.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();

                let mut target_q = self.network.forward(&state);
                if won {
                    target_q[action] = reward;
                } else {
                    target_q[action] = reward + discount * max_next_q;
                }

                self.network.back(&state, &target_q);

                if won {
                    break;
                }

                current_player = !current_player;
            }

            epsilon = (epsilon * epsilon_decay).max(epsilon_min);

            if episode % 1000 == 0 {
                println!("Episode {} done. Epsilon = {:.3}", episode, epsilon);
            }
        }
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
