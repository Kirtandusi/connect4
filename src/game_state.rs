use rand::Rng;
use term;

pub struct GameState {
    pub board: Vec<Vec<usize>>,
    pub exclude: Vec<usize>,
}

impl GameState {
    // Create a new game state with an empty board
    pub fn new() -> GameState {
        GameState {
            board: vec![vec![0; 7]; 6],
            exclude: Vec::new(),
        }
    }
    pub fn check_if_full(&self, column: usize) -> bool {
        self.board[0][column] != 0
    }

    pub fn is_not_full(&self) -> bool {
        for i in 0..7 {
            for j in 0..6 {
                if self.board[j][i] == 0 {
                    return true;
                }
            }
        }
        false
    }

    pub fn top(&self, column: usize) -> usize {
        for i in (0..6).rev() {
            if self.board[i][column] == 0 {
                return i;
            }
        }
        6
    }

    pub fn play_move(&mut self, column: usize, side: bool) {
        let top_of_column = self.top(column);
        if top_of_column == 6 {
            println!("Column is full, please choose another column");
        } else {
            if side {
                self.board[top_of_column][column] = 1; // Player move
            } else {
                self.board[top_of_column][column] = 2; // CPU move
            }
        }
    }

    pub fn cpu_random(&mut self) {
        if self.exclude.len() == 7 {
            println!("Board is full, game is a tie!");
            return;
        }
        let mut rng = rand::thread_rng();
        let valid_columns: Vec<usize> = (0..7).filter(|&x| !self.exclude.contains(&x)).collect();
        if valid_columns.is_empty() {
            println!("No valid columns left, game over!");
            return;
        }
        let column = valid_columns[rng.gen_range(0..valid_columns.len())];
        if !self.check_if_full(column) {
            self.play_move(column, false); // CPU move
        } else {
            self.exclude.push(column);
            self.cpu_random();
        }
    }

    pub fn board_to_string(&self) {
        let mut terminal = term::stdout().unwrap();
        for i in 0..6 {
            for j in 0..7 {
                if self.board[i][j] == 1 {
                    terminal.fg(term::color::YELLOW).unwrap();
                } else if self.board[i][j] == 2 {
                    terminal.fg(term::color::RED).unwrap();
                } else {
                    terminal.fg(term::color::WHITE).unwrap();
                }
                print!("{} ", self.board[i][j]);
            }
            terminal.fg(term::color::WHITE).unwrap();
            println!();
        }
    }

    pub fn check_for_win(&self) -> bool {
        false
    }
}
