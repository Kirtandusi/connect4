use crate::game_state::GameState;
use crate::player::Player;

//implements Alpha Beta Pruning, moving ordering, and transposition table
//https://tromp.github.io/c4/fhour.html
//http://blog.gamesolver.org
//when its user's turn, they will try to maximize score of next possible positions
//opponent plays score that maximizes their chances, minimizes users
pub struct MinMaxPlayer;
impl MinMaxPlayer {
    fn evaluate_board(&self, gamestate: &GameState) -> isize {
        //arbitrary scores.
        const WEIGHTS: [[isize; 7]; 6] = [
            [3, 4, 5, 7, 5, 4, 3],
            [4, 6, 8, 10, 8, 6, 4],
            [5, 7, 11, 13, 11, 7, 5],
            [5, 7, 11, 13, 11, 7, 5],
            [4, 6, 8, 10, 2, 1, 0],
            [3, 4, 5, 7, 5, 4, 3],
        ];

        //calculate total score of the board
        let mut score: isize = 0;
        for i in 0..6 {
            for j in 0..7 {
                if gamestate.board[i][j] == 1 {
                    score -= WEIGHTS[i][j] // Human Player
                } else if gamestate.board[i][j] == 2 {
                    score += WEIGHTS[i][j]; // AI Player
                }
            }
        }
        score
    }
    
    //all available moves to the AI
    fn generate_moves(&self, gamestate: &GameState) -> Vec<usize> {
        let mut moves: Vec<usize> = (0..7)
            .filter(|&col| !gamestate.check_if_full(col))
            .collect();

        // Sort moves to prioritize center columns
        moves.sort_by(|a, b| {
            let center_dist_a = (3 as isize - *a as isize).abs();
            let center_dist_b = (3 as isize - *b as isize).abs();
            center_dist_a.cmp(&center_dist_b)
        });

        moves
    }

    // Main minimax function to find the best move
    fn minimax(&self, gamestate: &mut GameState, depth: usize, maximizing: bool) -> usize {
        let valid_moves = self.generate_moves(gamestate);
        if valid_moves.is_empty() {
            return 0; // Default to column 0 if no valid moves (shouldn't happen)
        }
        
        let alpha = isize::MIN;
        let beta = isize::MAX;

        if maximizing {
            self.maximize(gamestate, depth, alpha, beta)
        } else {
            self.minimize(gamestate, depth, alpha, beta)
        }
    }
    
    fn maximize(&self, gamestate: &mut GameState, depth: usize, alpha: isize, beta: isize) -> usize {
        let valid_moves = self.generate_moves(gamestate);
        
        if depth == 0 || valid_moves.is_empty() || gamestate.check_for_win() {
            return valid_moves.get(0).copied().unwrap_or(0);
        }
        
        let mut best_score = isize::MIN;
        let mut best_move = valid_moves[0];
        let mut current_alpha = alpha;

        for &col in &valid_moves {
            let mut new_state = gamestate.clone();
            new_state.play_move(col, false); // AI's move
            
            let score = self.evaluate_position(&new_state, depth - 1, current_alpha, beta, false);
            
            if score > best_score {
                best_score = score;
                best_move = col;
            }
            
            current_alpha = current_alpha.max(best_score);
            if current_alpha >= beta {
                break; // Beta cutoff
            }
        }
        
        best_move
    }
    
    fn minimize(&self, gamestate: &mut GameState, depth: usize, alpha: isize, beta: isize) -> usize {
        let valid_moves = self.generate_moves(gamestate);
        
        if depth == 0 || valid_moves.is_empty() || gamestate.check_for_win() {
            return valid_moves.get(0).copied().unwrap_or(0);
        }
        
        let mut best_score = isize::MAX;
        let mut best_move = valid_moves[0];
        let mut current_beta = beta;

        for &col in &valid_moves {
            let mut new_state = gamestate.clone();
            new_state.play_move(col, true); // Human's move
            
            let score = self.evaluate_position(&new_state, depth - 1, alpha, current_beta, true);
            
            if score < best_score {
                best_score = score;
                best_move = col;
            }
            
            current_beta = current_beta.min(best_score);
            if current_beta <= alpha {
                break; // Alpha cutoff
            }
        }
        
        best_move
    }
    
    fn evaluate_position(&self, gamestate: &GameState, depth: usize, alpha: isize, beta: isize, maximizing: bool) -> isize {
        // Terminal conditions
        if depth == 0 || self.generate_moves(gamestate).is_empty() || gamestate.check_for_win() {
            return self.evaluate_board(gamestate);
        }
        
        if maximizing {
            let mut best_score = isize::MIN;
            let mut current_alpha = alpha;

            for &col in &self.generate_moves(gamestate) {
                let mut new_state = gamestate.clone();
                new_state.play_move(col, false); // AI's move

                let score = self.evaluate_position(&new_state, depth - 1, current_alpha, beta, false);
                best_score = best_score.max(score);
                current_alpha = current_alpha.max(best_score);
                
                if beta <= current_alpha {
                    break; // Beta cutoff
                }
            }
            
            return best_score;
        } else {
            let mut best_score = isize::MAX;
            let mut current_beta = beta;

            for &col in &self.generate_moves(gamestate) {
                let mut new_state = gamestate.clone();
                new_state.play_move(col, true); // Human's move

                let score = self.evaluate_position(&new_state, depth - 1, alpha, current_beta, true);
                best_score = best_score.min(score);
                current_beta = current_beta.min(best_score);
                
                if current_beta <= alpha {
                    break; // Alpha cutoff
                }
            }
            
            return best_score;
        }
    }
}

impl Player for MinMaxPlayer {
    fn make_move(&mut self, gamestate: &mut GameState) {
        let valid_moves = self.generate_moves(gamestate);
        if valid_moves.is_empty() {
            return;
        }
        
        // Find the best move using minimax, with depth 3 and maximizing (AI's turn)
        let best_move = self.minimax(gamestate, 3, true);
        gamestate.play_move(best_move, false); // AI makes its move (false = AI)
    }
}