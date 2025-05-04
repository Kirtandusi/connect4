use crate::game_state::GameState;
use crate::player::Player;

pub struct MinMaxPlayer {
    player: bool,
}

impl MinMaxPlayer {
    pub fn new(player: bool) -> Self {
        MinMaxPlayer { player }
    }
    fn evaluate_board(&self, gamestate: &GameState) -> isize {
        // Heuristic weights matrix (same as before)
        const WEIGHTS: [[isize; 7]; 6] = [
            [3, 4, 5, 7, 5, 4, 3],
            [4, 6, 8, 10, 8, 6, 4],
            [5, 7, 11, 13, 11, 7, 5],
            [5, 7, 11, 13, 11, 7, 5],
            [4, 6, 8, 10, 8, 6, 4],
            [3, 4, 5, 7, 5, 4, 3],
        ];

        let mut score = 0;
        for (i, row) in gamestate.board.iter().enumerate() {
            for (j, &cell) in row.iter().enumerate() {
                match cell {
                    1 => score -= WEIGHTS[i][j], // Human player
                    2 => score += WEIGHTS[i][j], // AI player
                    _ => (),
                }
            }
        }
        score
    }

    fn minimax(
        &self,
        gamestate: &mut GameState,
        depth: usize,
        mut alpha: isize,
        mut beta: isize,
        maximizing: bool,
    ) -> isize {
        // Check for terminal states
        if gamestate.check_for_win() {
            return match gamestate.winner {
                2 => isize::MAX, // player 2 wins
                1 => isize::MIN, // player 1 wins
                _ => 0,          // draw but should never happen
            };
        }

        // Then check depth limit
        if depth == 0 {
            return self.evaluate_board(gamestate);
        }

        let valid_moves = self.generate_moves(gamestate);

        if maximizing {
            let mut max_score = isize::MIN;
            for &col in &valid_moves {
                let mut new_state = gamestate.clone();
                new_state.play_move(col, self.player);

                let score = self.minimax(&mut new_state, depth - 1, alpha, beta, false);
                max_score = max_score.max(score);
                alpha = alpha.max(max_score);

                if alpha >= beta {
                    //pruning
                    break; // Beta cutoff
                }
            }
            max_score
        } else {
            let mut min_score = isize::MAX;
            for &col in &valid_moves {
                let mut new_state = gamestate.clone();
                new_state.play_move(col, true);

                let score = self.minimax(&mut new_state, depth - 1, alpha, beta, true);
                min_score = min_score.min(score);
                beta = beta.min(min_score);

                if beta <= alpha {
                    break; // Alpha cutoff
                }
            }
            min_score
        }
    }

    fn find_best_move(&self, gamestate: &GameState, depth: usize) -> usize {
        let mut best_score = isize::MIN;
        let mut best_move = 0;
        let alpha = isize::MIN; //negative infinity
        let beta = isize::MAX; //positive infinity

        for &col in &self.generate_moves(gamestate) {
            let mut new_state = gamestate.clone();
            new_state.play_move(col, false);

            let score = self.minimax(&mut new_state, depth - 1, alpha, beta, false);
            if score > best_score || (score == best_score && col == 3) {
                // Prefer center column
                best_score = score;
                best_move = col;
            }
        }

        best_move
    }
}

impl Player for MinMaxPlayer {
    fn make_move(&mut self, gamestate: &mut GameState) {
        let best_move = self.find_best_move(gamestate, 5); // 5 layers of depth
        gamestate.play_move(best_move, self.player);
    }
    fn get_name(&self) -> &str {
        "Minimax Player"
    }
}
