use std::io::{self, Write};
use crate::game_state::GameState;
use crate::player::Player;

pub struct HumanPlayer {
    player: bool,
}

impl HumanPlayer {
    pub fn new(player: bool) -> Self {
        HumanPlayer { player }
    }

    fn read_input(&self, prompt: &str) -> String {
        print!("{}", prompt);
        io::stdout().flush().unwrap();

        let mut input = String::new();
        io::stdin().read_line(&mut input).expect("Failed to read input");
        input.trim().to_string()
    }
}

impl Player for HumanPlayer {
    fn make_move(&mut self, game_state: &mut GameState) {
        println!("Current game state:");
        game_state.board_to_string();

        loop {
            let input = self.read_input("Enter your move (column 1-7): ");
            match input.parse::<usize>() {
                Ok(column) if column <= 7 && column >= 1 => {
                    if !game_state.check_if_full(column-1) {
                        game_state.play_move(column-1, self.player);
                        break;
                    } else {
                        println!("Column {} is full. Please try another column.", column);
                    }
                },
                Ok(_) => println!("Please enter a number between 1 and 7."),
                Err(_) => println!("Please enter a valid number."),
            }
        }
    }

    fn get_name(&self) -> &str {
        "Human Player"
    }
}