use rand::prelude::IndexedRandom;
use crate::game_state::GameState;
use crate::player::Player;
use rand::Rng;
use crate::minimax_player::MinMaxPlayer;
use crate::neuralnetwork::neural_network::{Layer, NeuralNetwork};
use crate::neuralnetwork::neuron::Neuron;
use crate::random_player::RandomPlayer;

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
                Neuron::new(weights, bias, Neuron::leaky_relu_activation, Neuron::leaky_relu_derivative)
            })
            .collect();

        // Add a second hidden layer for better performance
        let hidden_layer2: Layer = (0..hidden_size)
            .map(|_| {
                let weights: Vec<f64> = (0..hidden_size)
                    .map(|_| rng.random_range(-0.5..0.5))
                    .collect();
                let bias = rng.random_range(-0.5..0.5);
                Neuron::new(weights, bias, Neuron::identity, Neuron::identity_derivative)
            })
            .collect();

        let output_layer: Layer = (0..output_size)
            .map(|_| {
                let weights: Vec<f64> = (0..hidden_size) // Connect to second hidden layer
                    .map(|_| rng.random_range(-0.5..0.5)) // Fixed: was random_range
                    .collect();
                let bias = rng.random_range(-0.5..0.5); // Fixed: was random_range
                Neuron::new(weights, bias, Neuron::identity, Neuron::identity_derivative)
            })
            .collect();

        let network = NeuralNetwork::new(vec![hidden_layer, hidden_layer2, output_layer], 0.001);

        Self { player, network }
    }
    /*
        the neural network will train itself.
     */
    pub fn train_generalized(&mut self, episodes: usize) {
        let mut rng = rand::rng();
        let discount: f32 = 0.95; // Slightly reduced discount factor
        let mut epsilon = 1.0;
        let epsilon_min = 0.01; // Increased minimum exploration rate
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
            let mut opponent: Box<dyn Player> = if roll < 0.33 {
                Box::new(RandomPlayer::new(!self.player))
            } else if roll < 0.66 {
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

                        //apply a mask to get rid of values in full columns.
                        let best_action = valid_moves.iter()
                            .max_by(|&&a, &&b| q_values[a].partial_cmp(&q_values[b]).unwrap_or(std::cmp::Ordering::Equal))
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
                0.0 // Draw or unfinished (slight positive reward to encourage longer games)
            };

            for i in 0..game_history.len() {
                let (ref state, action) = game_history[i];
                let reward_for_this_step = if i == game_history.len() - 1 {
                    reward // terminal reward
                } else {
                    0.0 // intermediate moves get 0 reward unless you implement shaping
                };

                // next state (or 0 if terminal)
                let next_q_max = if i + 1 < game_history.len() {
                    let (next_state, _) = &game_history[i + 1];
                    let q_next = self.network.forward(next_state);
                    if q_next.is_empty() {
                        0.0
                    } else { //q values are not being updated, as last value is always picked.
                        *q_next.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap()
                    }
                } else {
                    0.0
                };

                let target = reward_for_this_step as f64 + (discount as f64) * next_q_max;

                let mut q_values = self.network.forward(state);

                q_values[action] = target;

                self.network.back(state, &q_values);
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
        let best_action = match valid_moves.iter()
            .max_by(|&&a, &&b| q_values[a].partial_cmp(&q_values[b]).unwrap()) {
            Some(a) => *a,
            None => {
                println!("valid moves empty");
                return;
            }
        };



        if !game_state.get_valid_moves().contains(&best_action) {
            println!("Training: selected full column {}! Skipping turn.", best_action);
            return;
        }

        game_state.play_move(best_action, self.player);
    }
    fn get_name(&self) -> &str {
        "Neural Net Player"
    }
}
