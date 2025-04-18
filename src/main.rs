use std::{io, process};

mod game_state;
mod player;
mod random_player;
mod minimax_player;
mod neuralnet_player;
mod human;

use game_state::GameState;
use player::Player;
use random_player::RandomPlayer;
use minimax_player::MinMaxPlayer;
use neuralnet_player::NeuralNetPlayer;
use human::HumanPlayer;

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
            return Some(Box::new(NeuralNetPlayer::new(side)));
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
    while board.is_not_full() {
        player1.make_move(&mut board);
        board.board_to_string();
        if board.check_for_win() {
            println!("Player 1 wins!");
            return;
        }
        player2.make_move(&mut board);
        board.board_to_string();
        if board.check_for_win() {
            println!("Player 2 wins!");
        }
    }
    println!("Game over.");
}
fn simulate(player1: &mut dyn Player, player2: &mut dyn Player) {
    println!("{}", format!("Starting Connect 4 simulation between {} and {}", player1.get_name(), player2.get_name()));
    let mut board = GameState::new();
    board.board_to_string(); // Print the initial state of the board

    while board.is_not_full() {
        // Player 1's move
        player1.make_move(&mut board);
        println!("{}", format!("{} {}", player1.get_name(), "'s move:"));
        board.board_to_string();

        if board.check_for_win() {
            println!("{}", format!("{} {}", player1.get_name(), "wins!"));
            return;
        }

        if !board.is_not_full() {
            break; // Break if the board is full after Player 1's move
        }

        // Player 2's move
        player2.make_move(&mut board);
        println!("{}", format!("{} {}", player2.get_name(), "'s move:"));
        board.board_to_string();

        if board.check_for_win() {
            println!("{}", format!("{} {}", player2.get_name(), "wins!"));
            return;
        }
    }

    println!("Game over. It's a draw!");
}
