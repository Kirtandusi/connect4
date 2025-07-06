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
                    .map(|_| rng.random_range(-0.5..0.5))
                    .collect();
                let bias = rng.random_range(-0.5..0.5);
                Neuron::new(weights, bias, Neuron::leaky_relu_activation, Neuron::leaky_relu_derivative)
            })
            .collect();

        // SECOND hidden layer, same as first but uses different activation fn to stop gradient issues
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
                    .map(|_| rng.random_range(-0.5..0.5))
                    .collect();
                let bias = rng.random_range(-0.5..0.5);
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

        let mut random_chance = 0.7;
        let mut self_play_chance = 0.25;
        for episode in 0..episodes {

            //changes increase to more minimax as episodes increase.
            if episode == 5000 {
                random_chance = 0.4;
                self_play_chance = 0.8;
            } else if episode == 15000 {
                random_chance = 0.2;
                self_play_chance = 0.5;
            } else if episode == 25000 {
                random_chance = 0.1;
                self_play_chance = 0.2;
            }

            let mut game = GameState::new();
            let mut current_player = true;

            // Choose opponent: 50% random, 30% self-play, 20% Minimax //CHANGE THESE VALUES?
            let roll: f64 = rng.random();
            let mut opponent: Box<dyn Player> = if roll < random_chance {
                Box::new(RandomPlayer::new(!self.player))
            } else if roll < self_play_chance {
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

            //state action reward tracking
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
                    // exploration or explotation
                    let action = if rng.random::<f64>() < epsilon {
                        // Exploration
                        *valid_moves.choose(&mut rng).unwrap()
                    } else {
                        // exploitation, go by q values
                        let q_values = self.network.forward(&state);

                        //apply a mask to get rid of values in full columns. [with help of ai]
                        let best_action = valid_moves.iter()
                            .max_by(|&&a, &&b| {
                                let q_a = q_values.get(a).unwrap_or(&0.0);
                                let q_b = q_values.get(b).unwrap_or(&0.0);
                                q_a.partial_cmp(q_b).unwrap_or(std::cmp::Ordering::Equal)
                            });

                        match best_action {
                            Some(&action) => action,
                            None => *valid_moves.choose(&mut rng).unwrap() // Fallback to random
                        }
                    };

                    //store state action pair
                    game_history.push((state, action));

                    // Make the actual move
                    let _  = game.play_move(action, current_player);

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

            // assigning rewards for events
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
                    0.0
                };

                // next state (or 0 if terminal)
                let next_q_max = if i + 1 < game_history.len() {
                    let (next_state, _) = &game_history[i + 1];
                    let q_next = self.network.forward(next_state);
                    if q_next.is_empty() {
                        0.0
                    } else { //q values are not being updated, as last value is always picked.
                        *q_next.iter()
                            .filter(|x| !x.is_nan())
                            .max_by(|a, b| a.partial_cmp(b).unwrap())
                            .unwrap_or(&0.0)
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

            // for progress debugging and reporting
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

        println!("Training complete.");
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

        let best_action = valid_moves.iter()
            .max_by(|&&a, &&b| {
                let q_a = q_values.get(a).unwrap_or(&0.0); //BIG CHANGE HERE, use unwrap or to prevent panic.
                let q_b = q_values.get(b).unwrap_or(&0.0);
                q_a.partial_cmp(q_b).unwrap_or(std::cmp::Ordering::Equal)
            });

        let best_action = match best_action {
            Some(&action) => action,
            None => {
                println!("Could not determine best action, using first valid move");
                valid_moves[0]
            }
        };

        // Double-check that the action is still valid
        if !game_state.get_valid_moves().contains(&best_action) {
            println!("Selected action {} is no longer valid! Using first available move.", best_action);
            let current_valid = game_state.get_valid_moves();
            if !current_valid.is_empty() {
                let _ = game_state.play_move(current_valid[0], self.player);
            } else {
                println!("No valid moves available!");
            }
            return;
        }

        let _ = game_state.play_move(best_action, self.player);
    }

    fn get_name(&self) -> &str {
        "Neural Net Player"
    }
}
