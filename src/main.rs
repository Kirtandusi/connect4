use std::io;
mod game_state;
use game_state::GameState;

fn main() {
    input();
}

fn input() {
    println!("Welcome to Connect 4. User goes first. Player moves get marked with a 1, and CPU moves get marked with a 2.");
    let mut loss = false;

    // Initialize the board using GameState::new()
    let mut board = GameState::new();
    board.board_to_string(); // No need for `self` here

    let mut input_str: String = String::new();
    while !loss && board.is_not_full() { // input cycle
        input_str.clear();
        println!("Please choose a column to drop your piece. Input range is 1-7");
        io::stdin().read_line(&mut input_str).expect("Failed to read input");
       
        let mut column: usize = match input_str.trim().parse() {
            Ok(num) => num,
            Err(_) => {
                println!("Invalid input, please try again");
                continue;
            }
        };
        
        if column < 1 || column > 7 {
            println!("Invalid input, please try again");
            continue;
        }
        column = column - 1; // Convert to 0-based index
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

        board.cpu_random(); // CPU move
        
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
