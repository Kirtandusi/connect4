use crate::game_state::GameState;
use crate::player::Player;

pub struct MinMaxPlayer;

impl MinMaxPlayer {
    // Evaluate the board state (heuristic function)
    fn evaluate_board(&self, gamestate: &GameState) -> isize {
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

    // Generate all valid moves (columns that are not full)
    fn generate_moves(&self, gamestate: &GameState) -> Vec<usize> {
        (0..7).filter(|&col| !gamestate.check_if_full(col)).collect()
    }

    // Main minimax function with Alpha-Beta pruning
    fn minimax(
        &self,
        gamestate: &mut GameState,
        depth: usize,
        mut alpha: isize,
        mut beta: isize,
        maximizing: bool,
    ) -> isize {
        // Check for terminal states (win, lose, or draw)
        // if gamestate.check_for_win() {
        //     return if gamestate.winner == 2 {
        //         isize::MAX // AI wins
        //     } else if gamestate.winner == 1 {
        //         isize::MIN // Human wins
        //     } else {
        //         0 // Draw
        //     };
        // }

        // If depth limit is reached, return the heuristic evaluation
        if depth == 0 {
            //return self.evaluate_board(gamestate);
            return 99999999;
        }

        // Generate all valid moves
        let valid_moves = self.generate_moves(gamestate);

        if maximizing {
            // Maximizing player (AI)
            let mut max_score = isize::MIN;
            for &col in &valid_moves {
                let mut new_state = gamestate.clone();
                new_state.play_move(col, false); // AI's move

                let score = self.minimax(&mut new_state, depth - 1, alpha, beta, false);
                max_score = max_score.max(score);
                alpha = alpha.max(max_score);

                // Alpha-Beta pruning
                if alpha >= beta {
                    break;
                }
            }
            max_score
        } else {
            // Minimizing player (Human)
            let mut min_score = isize::MAX;
            for &col in &valid_moves {
                let mut new_state = gamestate.clone();
                new_state.play_move(col, true); // Human's move

                let score = self.minimax(&mut new_state, depth - 1, alpha, beta, true);
                min_score = min_score.min(score);
                beta = beta.min(min_score);

                // Alpha-Beta pruning
                if beta <= alpha {
                    break;
                }
            }
            min_score
        }
    }

    // Find the best move for the AI
    fn find_best_move(&self, gamestate: &GameState, depth: usize) -> usize {
        let mut best_score = isize::MIN;
        let mut best_move = 0; // Default to column 0 if no valid moves
        let alpha = isize::MIN;
        let beta = isize::MAX;

        for &col in &self.generate_moves(gamestate) {
            let mut new_state = gamestate.clone();
            new_state.play_move(col, false); // AI's move

            let score = self.minimax(&mut new_state, depth - 1, alpha, beta, false);
            if score > best_score {
                best_score = score;
                best_move = col;
            }
        }

        best_move
    }
}

impl Player for MinMaxPlayer {
    fn make_move(&mut self, gamestate: &mut GameState) {
        let best_move = self.find_best_move(gamestate, 5); // Adjust depth as needed
        gamestate.play_move(best_move, false); // AI makes its move
    }
}