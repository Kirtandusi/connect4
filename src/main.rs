use std::{io, process};
use crate::neuralnetwork::NeuralNetPlayer;
mod game_state;
mod human_player;
mod minimax_player;
mod neuralnetwork;
mod player;
mod random_player;

use game_state::GameState;
use human_player::HumanPlayer;
use minimax_player::MinMaxPlayer;
use player::Player;
use random_player::RandomPlayer;

fn main() {
    println!("Welcome to Connect-4!");
    input();
}
fn choose_player1(side: bool) -> Option<Box<dyn Player>> {
    println!("1. Human Player");
    println!("2. Random-Placement Player");
    println!("3. Min-Max Player");
    println!("4. Neural Network Player");
    println!("5. Quit");
    let mut choice = String::new();
    io::stdin()
        .read_line(&mut choice)
        .expect("Failed to read input");
    match choice.trim() {
        "1" => {
            println!("You have selected Human Player.");
            return Some(Box::new(HumanPlayer::new(side)));
        }
        "2" => {
            println!("You have selected Random Player.");
            return Some(Box::new(RandomPlayer::new(side)));
        }
        "3" => {
            println!("You have selected Min-Max Player.");
            return Some(Box::new(MinMaxPlayer::new(side)));
        }
        "4" => {
            println!("You have selected Neural Network Player.");
            println!("Training AI...");
            let trained_player = train_neural_net(side);
            return Some(Box::new(trained_player));
        }
        "5" => {
            println!("Exiting game.");
            process::exit(0);
        }
        _ => {
            println!("Invalid choice. Please select a valid option (1-5).");
        }
    }
    None
}
fn input() {
    println!("Choose Player 1.");
    let p1_options = choose_player1(true);
    println!("Choose Player 2.");
    let p2_options = choose_player1(false);
    if p1_options.is_none() || p2_options.is_none() {
        return;
    }
    let mut player1 = p1_options.unwrap();
    let mut player2 = p2_options.unwrap();
    println!(
        "Player 1 goes first. Player 1 get marked with a 1, and Player 2 get marked with a 2."
    );
    // Initialize the board using GameState::new()
    let mut board = GameState::new();
    board.board_to_string(); // No need for `self` here
    println!();
    while board.is_not_full() {
        player1.make_move(&mut board);
        board.board_to_string();
        println!();
        if board.check_for_win() {
            println!("Player 1 wins!");
            return;
        }
        player2.make_move(&mut board);
        board.board_to_string();
        println!();
        if board.check_for_win() {
            println!("Player 2 wins!");
            return;
        }
    }
    println!("Game over.");
}

fn train_neural_net(side: bool) -> NeuralNetPlayer {
    // let path = "trained_network.json";
    // if !retrain && Path::new(path).exists() {
    //     println!("Loading trained network from disk...");
    //    // let network = load_network(path);
    //     return NeuralNetPlayer { player: side, network };
    // }

   // println!("No saved network found. Training from scratch...");
    let mut ai_player = NeuralNetPlayer::new(side);
    ai_player.train_generalized(30000);
   // save_network(&ai_player.network, path);
    ai_player
}

// fn save_network(network: &NeuralNetwork, path: &str) {
//     let file = File::create(path).expect("Failed to create file");
//     let writer = BufWriter::new(file);
//     serde_json::to_writer(writer, network).expect("Failed to write network to file");
// }
//
// fn load_network(path: &str) -> NeuralNetwork {
//     let file = File::open(path).expect("Failed to open file");
//     let reader = BufReader::new(file);
//     let mut network: NeuralNetwork = serde_json::from_reader(reader).expect("Failed to read network");
//
//     // Reattach activation functions (ReLU)
//     for layer in &mut network.layers {
//         for neuron in layer {
//             neuron.activation = Neuron::relu_activation;
//         }
//     }
//
//     network
// }




