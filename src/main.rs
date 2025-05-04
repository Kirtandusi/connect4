use std::{io, process};

mod connect4_env;
mod game_state;
mod human_player;
mod minimax_player;
mod neuralnet_player;
mod player;
mod random_player;

use crate::connect4_env::Connect4Env;
use game_state::GameState;
use human_player::HumanPlayer;
use minimax_player::MinMaxPlayer;
use neuralnet_player::NeuralNetPlayer;
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
    let mut env = Connect4Env::new(side);
    let mut ai_player = NeuralNetPlayer::new(side);
    ai_player.network.train(&mut env);
    ai_player
}
