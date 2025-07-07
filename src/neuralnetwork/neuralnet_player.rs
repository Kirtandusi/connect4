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
        let discount: f64 = 0.95;
        let mut epsilon = 1.0;
        let epsilon_min = 0.01;
        let epsilon_decay = 0.9995;
        let mut win_count = 0;
        let mut loss_count = 0;
        let mut draw_count = 0;
        let mut best_win_rate = 0.0;

        // Keep best model for smarter self-play
        let mut best_network = self.network.clone();

        println!("Starting training for {} episodes...", episodes);

        for episode in 0..episodes {
            // Dynamically adjust opponent mix
            let (random_chance, self_play_chance) = if episode >= 20000 {
                (0.05, 0.15) // 80% MinMax
            } else if episode >= 15000 {
                (0.2, 0.5)
            } else if episode >= 5000 {
                (0.4, 0.4)
            } else {
                (0.7, 0.25)
            };

            let mut game = GameState::new();
            let mut current_player = true;

            // Opponent selection
            let roll: f64 = rng.random();
            let mut opponent: Box<dyn Player> = if roll < random_chance {
                Box::new(RandomPlayer::new(!self.player))
            } else if roll < random_chance + self_play_chance {
                Box::new(NeuralNetPlayer {
                    player: !self.player,
                    network: best_network.clone(),
                })
            } else {
                Box::new(MinMaxPlayer::new(!self.player))
            };

            let mut game_history: Vec<(Vec<f64>, usize)> = Vec::new();
            let mut player_won = false;
            let mut opponent_won = false;

            while game.is_not_full() && !player_won && !opponent_won {
                if current_player == self.player {
                    let state = game.to_input_vector();
                    let valid_moves = game.get_valid_moves();
                    if valid_moves.is_empty() {
                        break;
                    }

                    let action = if rng.random::<f64>() < epsilon {
                        *valid_moves.choose(&mut rng).unwrap()
                    } else {
                        let q_values = self.network.forward(&state);
                        *valid_moves.iter()
                            .max_by(|&&a, &&b| {
                                let q_a = q_values.get(a).unwrap_or(&0.0);
                                let q_b = q_values.get(b).unwrap_or(&0.0);
                                q_a.partial_cmp(q_b).unwrap_or(std::cmp::Ordering::Equal)
                            })
                            .unwrap_or_else(|| valid_moves.first().unwrap())
                    };

                    game_history.push((state, action));
                    let _ = game.play_move(action, current_player);

                    if game.check_for_win() {
                        player_won = true;
                        win_count += 1;
                    }
                } else {
                    opponent.make_move(&mut game);
                    if game.check_for_win() {
                        opponent_won = true;
                        loss_count += 1;
                    }
                }

                current_player = !current_player;
            }

            if !player_won && !opponent_won && !game.is_not_full() {
                draw_count += 1;
            }

            // Final reward
            let final_reward = if player_won {
                1.0
            } else if opponent_won {
                -1.0
            } else {
                0.0
            };

            // Q-learning update
            for i in 0..game_history.len() {
                let (ref state, action) = game_history[i];
                let reward = if i == game_history.len() - 1 {
                    final_reward
                } else {
                    0.0 // no reward shaping
                };

                let next_q_max = if i + 1 < game_history.len() {
                    let (next_state, _) = &game_history[i + 1];
                    let q_next = self.network.forward(next_state);
                    q_next.iter()
                        .filter(|x| !x.is_nan())
                        .cloned()
                        .fold(f64::NEG_INFINITY, f64::max)
                } else {
                    0.0
                };

                let target = reward + discount * next_q_max;
                let current_output = self.network.forward(state);
                let mut targets = current_output.clone();
                targets[action] = target;
                self.network.back(state, &targets);
            }

            // Epsilon decay
            epsilon = (epsilon * epsilon_decay).max(epsilon_min);

            // Epsilon recovery on performance collapse
            let total = win_count + loss_count + draw_count;
            if total >= 1000 {
                let win_rate = win_count as f64 / total as f64;
                if win_rate > best_win_rate {
                    best_win_rate = win_rate;
                    best_network = self.network.clone(); // promote to self-play opponent
                }

                if win_rate < 0.35 && epsilon <= epsilon_min + 1e-5 {
                    println!(
                        "Win rate dropped to {:.2}, increasing epsilon for recovery.",
                        win_rate
                    );
                    epsilon = 0.1;
                }
            }

            // Logging
            if (episode + 1) % 1000 == 0 || episode == 0 {
                println!(
                    "Episode {}/{} complete. Epsilon: {:.4}, Wins: {}, Losses: {}, Draws: {}",
                    episode + 1,
                    episodes,
                    epsilon,
                    win_count,
                    loss_count,
                    draw_count
                );
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
