pub struct GameState {
    pub board: Vec<Vec<usize>>,
    pub exclude: Vec<usize>,
    pub winner: usize,
    pub last_move: Option<usize>,
}
impl GameState {
    // Create a new game state with an empty board
    pub fn new() -> GameState {
        GameState {
            board: vec![vec![0; 7]; 6],
            exclude: Vec::new(),
            winner: 0,
            last_move: None,
        }
    }
    pub fn check_if_full(&self, column: usize) -> bool {
        self.board[0][column] != 0
    }

    pub fn is_not_full(&self) -> bool {
        for i in 0..6 {
            for j in 0..7 {
                if self.board[i][j] == 0 {
                    return true;
                }
            }
        }
        false
    }
    #[allow(dead_code)]
    pub fn get_last_move(&self) -> Option<usize> {
        self.last_move
    }
    pub fn to_input_vector(&self) -> Vec<f64> {
        let mut input = Vec::with_capacity(42);
        for row in 0..6 {
            for col in 0..7 {
                let value = match self.board[row][col] {
                    0 => 0.0,  // Empty
                    1 => 1.0,  // Player 1
                    2 => -1.0, // Player 2
                    _ => panic!("Invalid value in board!"),
                };
                input.push(value);
            }
        }
        input
    }
    pub fn top(&self, column: usize) -> usize {
        for i in (0..6).rev() {
            if self.board[i][column] == 0 {
                return i;
            }
        }
        6
    }
    /*
    takes mutable reference because it must change the gamestate.
     */
    pub fn play_move(&mut self, column: usize, side: bool) {
        self.last_move = Some(column);
        let top_of_column = self.top(column);
        if top_of_column == 6 {
            println!("Column is full, please choose another column");
        } else if side {
            self.board[top_of_column][column] = 1; // Player 1
        } else {
            self.board[top_of_column][column] = 2; // Player 2
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
    pub fn check_for_win(&mut self) -> bool {
        let length = self.board.len();
        let width = self.board[0].len();

        //horizontal checks
        for i in 0..length {
            let mut count_1: usize = 0;
            let mut count_2: usize = 0;
            //horizontal checks
            for j in 0..width {
                match self.board[i][j] {
                    1 => {
                        count_1 += 1;
                        count_2 = 0;
                    }
                    2 => {
                        count_2 += 1;
                        count_1 = 0;
                    }
                    _ => {
                        count_1 = 0;
                        count_2 = 0;
                    }
                }
                if count_2 == 4 {
                    self.winner = 2;
                    return true;
                } else if count_1 == 4 {
                    self.winner = 1;
                    return true;
                }
            }
        }

        //vertical checks
        for i in 0..width {
            let mut count_1: usize = 0;
            let mut count_2: usize = 0;
            for j in 0..length {
                match self.board[j][i] {
                    1 => {
                        count_1 += 1;
                        count_2 = 0;
                    }
                    2 => {
                        count_2 += 1;
                        count_1 = 0;
                    }
                    _ => {
                        count_1 = 0;
                        count_2 = 0;
                    }
                }
                if count_2 == 4 {
                    self.winner = 2;
                    return true;
                } else if count_1 == 4 {
                    self.winner = 1;
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
                    if player == 1 {
                        self.winner = 1;
                    } else {
                        self.winner = 2;
                    }
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
                    if player == 1 {
                        self.winner = 1;
                    } else {
                        self.winner = 2;
                    }
                    return true;
                }
            }
        }
        false
    }
    /*
       This is basically a get successors function. Critical for uniformed search
    */
    pub fn get_valid_moves(&self) -> Vec<usize> {
        (0..7).filter(|&col| !self.check_if_full(col)).collect()
    }
}
impl Clone for GameState {
    fn clone(&self) -> Self {
        Self {
            board: self.board.clone(),
            exclude: self.exclude.clone(),
            winner: self.winner.clone(),
            last_move: self.last_move.clone(),
        }
    }
}
