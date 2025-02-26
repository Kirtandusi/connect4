use std::io;

mod game_state;
mod player;
mod random_player;

use game_state::GameState;
use player::Player;
use random_player::RandomPlayer;

fn main() {
    input();
}
fn choose() -> Option<Box<dyn Player>> {
    println!("Welcome to Connect 4. Choose an opponent.");
    println!("1. Random-Placement Player");
    println!("2. Algorithmic Player");
    println!("3. Min-Max Player");
    println!("4. Neural Network Player");
    println!("5. Quit");
    let mut choice = String::new();
    io::stdin()
        .read_line(&mut choice)
        .expect("Failed to read input");
    match choice.trim() {
        "1" => {
            println!("You selected Random Player.");
            return Some(Box::new(RandomPlayer {}));
        }
        "2" => {
            println!("Algorithmic Player is not implemented yet.");
            return Some(Box::new(RandomPlayer {}));
        }
        "3" => {
            println!("Min-Max Player is not implemented yet.");
            return Some(Box::new(RandomPlayer {}));
        }
        "4" => {
            println!("Neural Network Player is not implemented yet.");
            return Some(Box::new(RandomPlayer {}));
        }
        "5" => {
            println!("Exiting game.");
            return Some(Box::new(RandomPlayer {}));
        }
        _ => {
            println!("Invalid choice. Please select a valid option (1-5).");
        }
    }
    None
}

fn input() {
    let options = choose();
    if options.is_none() {
        return;
    }
    let mut cpu = options.unwrap();
    println!(
        "User goes first. Player moves get marked with a 1, and CPU moves get marked with a 2."
    );
    let mut loss = false;

    // Initialize the board using GameState::new()
    let mut board = GameState::new();
    board.board_to_string(); // No need for `self` here

    let mut input_str: String = String::new();
    while !loss && board.is_not_full() {
        // input cycle
        input_str.clear();
        println!("Please choose a column to drop your piece. Input range is 1-7");
        io::stdin()
            .read_line(&mut input_str)
            .expect("Failed to read input");

        let mut column: usize = match input_str.trim().parse() {
            Ok(num) => num,
            Err(_) => {
                println!("Invalid input, please try again");
                continue;
            }
        };

        if !(1..=7).contains(&column) {
            println!("Invalid input, please try again");
            continue;
        }
        column -= 1; // Convert to 0-based index
        println!();

        if !board.check_if_full(column) {
            board.play_move(column, true); // Player move
        }

        if board.check_for_win() {
            println!("You win!");
            loss = true;
        }

        board.board_to_string(); // No need for `self` here
        println!("This is the new board after your move.");

        cpu.make_move(&mut board); // CPU move

        println!();
        board.board_to_string(); // No need for `self` here

        if board.check_for_win() {
            println!("CPU wins!");
            loss = true;
        } else {
            print!("This is the new board after the CPU's turn. ");
            if board.is_not_full() {
                println!("Your move.");
            }
            println!();
        }
    }
    println!("Game over");
}
