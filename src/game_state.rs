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
        } else if side {
            self.board[top_of_column][column] = 1; // Player move
        } else {
            self.board[top_of_column][column] = 2; // CPU move
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
        let length = self.board.len();
        let width = self.board[0].len();
        //vertical checks.
        for i in 0..length {
            let mut count_user: usize = 0;
            let mut count_cpu: usize = 0;
            for j in 0..width {
                match self.board[i][j] {
                    1 => {
                        count_user += 1;
                        count_cpu = 0;
                    }
                    2 => {
                        count_cpu += 1;
                        count_user = 0;
                    }
                    _ => {
                        count_user = 0;
                        count_cpu = 0;
                    }
                }
                if count_user == 4 || count_cpu == 4 {
                    return true;
                }
            }
        }
        //horizontal checks
        for i in 0..width {
            let mut count_user: usize = 0;
            let mut count_cpu: usize = 0;
            for j in 0..length {
                match self.board[j][i] {
                    1 => {
                        count_user += 1;
                        count_cpu = 0;
                    }
                    2 => {
                        count_cpu += 1;
                        count_user = 0;
                    }
                    _ => {
                        count_user = 0;
                        count_cpu = 0;
                    }
                }
                if count_user == 4 || count_cpu == 4 {
                    return true;
                }
            }
        }

        //diagonal checks
        //bottom left to top right
        for i in 0..3 {
            for j in 0..4 {
                let player = self.board[i][j];
                if player != 0
                    && self.board[i + 1][j + 1] == player
                    && self.board[i + 2][j + 2] == player
                    && self.board[i + 3][j + 3] == player
                {
                    return true;
                }
            }
        }
        //top left to bottom right
        for i in 3..6 {
            for j in 0..4 {
                let player = self.board[i][j];
                if player != 0
                    && self.board[i - 1][j + 1] == player
                    && self.board[i - 2][j + 2] == player
                    && self.board[i - 3][j + 3] == player
                {
                    return true;
                }
            }
        }
        false
    }
}
