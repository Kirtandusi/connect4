use rand::prelude::IndexedRandom;
use crate::game_state::GameState;
use crate::player::Player;
use rand::Rng;
use crate::minimax_player::MinMaxPlayer;
use crate::random_player::RandomPlayer;

#[derive(Clone)]
pub struct Neuron {
    weights: Vec<f64>,          // Weights for each input
    bias: f64,                  // Bias value
    pub(crate) activation: fn(f64) -> f64, // Activation function
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

#[derive(Clone)]
pub struct NeuralNetwork {
    pub(crate) layers: Vec<Layer>,
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
    calculates mean squared error between target and prediction. needed for debugging.
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
}
pub struct NeuralNetPlayer {
    pub(crate) player: bool,
    pub network: NeuralNetwork,
}

//deep Q learning implementation. Values are set to be random instead of 0. Will resolve naturally.
// learning rate implemented during back prop, not needed during Q learning.
impl NeuralNetPlayer {
    pub fn new(player: bool) -> Self {
        let input_size = 42; //connect 4 is 7x6
        let hidden_size = 32; //just a random number - CHANGE THIS MAYBE?
        let output_size = 7; //7 columns AI can move.
        let mut rng = rand::rng();

        //initialize hidden layer.
        //random weights, biases
        let hidden_layer: Layer = (0..hidden_size)
            .map(|_| {
                let weights: Vec<f64> = (0..input_size)
                    .map(|_| rng.random_range(-0.5..0.5)) // Fixed: was random_range
                    .collect();
                let bias = rng.random_range(-0.5..0.5); // Fixed: was random_range
                Neuron::new(weights, bias, Neuron::relu_activation)
            })
            .collect();

        // Add a second hidden layer for better performance
        let hidden_layer2: Layer = (0..hidden_size)
            .map(|_| {
                let weights: Vec<f64> = (0..hidden_size)
                    .map(|_| rng.random_range(-0.5..0.5))
                    .collect();
                let bias = rng.random_range(-0.5..0.5);
                Neuron::new(weights, bias, Neuron::relu_activation)
            })
            .collect();

        let output_layer: Layer = (0..output_size)
            .map(|_| {
                let weights: Vec<f64> = (0..hidden_size) // Connect to second hidden layer
                    .map(|_| rng.random_range(-0.5..0.5)) // Fixed: was random_range
                    .collect();
                let bias = rng.random_range(-0.5..0.5); // Fixed: was random_range
                Neuron::new(weights, bias, Neuron::relu_activation)
            })
            .collect();

        let network = NeuralNetwork::new(vec![hidden_layer, hidden_layer2, output_layer], 0.01);

        Self { player, network }
    }
    /*
        the neural network will train itself.
     */
    pub fn train_generalized(&mut self, episodes: usize) {
        let mut rng = rand::rng();
        let discount: f32 = 0.95; // Slightly reduced discount factor
        let mut epsilon = 1.0;
        let epsilon_min = 0.1; // Increased minimum exploration rate
        let epsilon_decay = 0.9995; // Slower decay

        println!("Starting training for {} episodes...", episodes);

        let mut win_count = 0;
        let mut loss_count = 0;
        let mut draw_count = 0;

        for episode in 0..episodes {
            let mut game = GameState::new();
            let mut current_player = true;

            // Choose opponent: 50% random, 30% self-play, 20% Minimax
            let roll: f64 = rng.random();
            let mut opponent: Box<dyn Player> = if roll < 0.5 {
                Box::new(RandomPlayer::new(!self.player))
            } else if roll < 0.8 {
                // Self-play with a clone of the current network
                let cloned_net = NeuralNetwork {
                    layers: self.network.layers.clone(),
                    learning_rate: self.network.learning_rate,
                };
                Box::new(NeuralNetPlayer {
                    player: !self.player,
                    network: cloned_net,
                })
            } else {
                Box::new(MinMaxPlayer::new(!self.player))
            };

            // FIXED: Added state-action-reward history tracking
            let mut game_history: Vec<(Vec<f64>, usize)> = Vec::new(); // State, action pairs
            let mut player_won = false;
            let mut opponent_won = false;

            while game.is_not_full() && !player_won && !opponent_won {
                if current_player == self.player {
                    // AI's turn
                    let state = game.to_input_vector();
                    let valid_moves = game.get_valid_moves();

                    if valid_moves.is_empty() {
                        break; // No valid moves available
                    }

                    // Decide move (explore or exploit)
                    let action = if rng.random::<f64>() < epsilon {
                        // Exploration: choose random valid move
                        *valid_moves.choose(&mut rng).unwrap()
                    } else {
                        // Exploitation: choose best move according to Q-values
                        let q_values = self.network.forward(&state);

                        // FIXED: Better handling of valid move selection
                        let best_action = valid_moves.iter()
                            .max_by(|&&a, &&b| {
                                q_values[a].partial_cmp(&q_values[b]).unwrap_or(std::cmp::Ordering::Equal)
                            })
                            .unwrap_or(&0);
                        *best_action
                    };

                    // FIXED: Store state-action pair for later batch training
                    game_history.push((state, action));

                    // Make the move
                    game.play_move(action, current_player);

                    // Check if player won
                    if game.check_for_win() {
                        player_won = true;
                        win_count += 1;
                    }
                } else {
                    // Opponent's turn
                    let _prev_state = game.clone();
                    opponent.make_move(&mut game);

                    // Check if opponent won
                    if game.check_for_win() {
                        opponent_won = true;
                        loss_count += 1;
                    }
                }

                // Switch players
                current_player = !current_player;
            }

            // If the game ended in a draw
            if !player_won && !opponent_won && !game.is_not_full() {
                draw_count += 1;
            }

            // FIXED: Proper terminal reward assignment
            let reward = if player_won {
                1.0 // Win
            } else if opponent_won {
                -1.0 // Loss
            } else {
                0.1 // Draw or unfinished (slight positive reward to encourage longer games)
            };

            // FIXED: Temporal difference learning implementation
            // Backward pass for each state-action pair with proper reward propagation
            for (i, (state, action)) in game_history.iter().enumerate() {
                // Calculate target Q-value with temporal difference - reward diminishes with distance from outcome
                let discounted_future_reward = reward * discount.powi((game_history.len() - i) as i32);

                // Get current Q-values
                let mut target_q = self.network.forward(state);

                // FIXED: Only update the Q-value for the chosen action
                // This implements the Q-learning update rule: Q(s,a) = r + γ·max_a'(Q(s',a'))
                target_q[*action] = discounted_future_reward as f64;

                // Train network
                self.network.back(state, &target_q);
            }

            // Decay epsilon
            epsilon = (epsilon * epsilon_decay).max(epsilon_min);

            // Progress reporting
            if (episode + 1) % 1000 == 0 || episode == 0 {
                println!(
                    "Episode {}/{} complete. Epsilon: {:.4}, Wins: {}, Losses: {}, Draws: {}",
                    episode + 1, episodes, epsilon, win_count, loss_count, draw_count
                );
                // Reset counters for next batch
                win_count = 0;
                loss_count = 0;
                draw_count = 0;
            }
        }

        println!("Training complete!");
    }
    #[allow(dead_code)]
    pub fn evaluate(&mut self, num_games: usize) -> (usize, usize, usize) {
        let mut wins = 0;
        let mut losses = 0;
        let mut draws = 0;

        // Create a MinMaxPlayer as opponent
        let mut opponent = MinMaxPlayer::new(!self.player);

        for _ in 0..num_games {
            let mut game = GameState::new();
            let mut current_player = true; // Player 1 starts

            while game.is_not_full() {
                if current_player == self.player {
                    self.make_move(&mut game);
                } else {
                    opponent.make_move(&mut game);
                }

                if game.check_for_win() {
                    if current_player == self.player {
                        wins += 1;
                    } else {
                        losses += 1;
                    }
                    break;
                }

                current_player = !current_player;
            }

            if !game.is_not_full() && !game.check_for_win() {
                draws += 1;
            }
        }

        (wins, losses, draws)
    }
}
impl Player for NeuralNetPlayer {
    fn make_move(&mut self, game_state: &mut GameState) {
        let input = game_state.to_input_vector();
        let q_values = self.network.forward(&input);

        let valid_moves = game_state.get_valid_moves();
        if valid_moves.is_empty() {
            println!("No valid moves found");
            return;
        }
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
