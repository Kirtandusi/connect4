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
    fn generate_moves(&self, gamestate: &mut GameState) -> Vec<usize> {
        (0..7)
            .filter(|&col| !gamestate.check_if_full(col))
            .collect()
    }
    fn minimax(
        &self,
        gamestate: &mut GameState,
        depth: usize,
        mut alpha: isize,
        mut beta: isize,
        maximizing: bool,
    ) -> (isize, usize) {
        //returns score of the board, then best column (for recursion)
        //alpha >= beta
        //base case
        if depth == 0 || !gamestate.is_not_full() || gamestate.check_for_win() {
            return (self.evaluate_board(gamestate), 0);
        }
        let valid_moves = self.generate_moves(gamestate);
        let mut best_col = valid_moves[0];

        if maximizing {
            let mut max_eval = isize::MIN;
            for &col in     &valid_moves {
                gamestate.play_move(col, false);
                let (eval, good_col) = self.minimax(gamestate, depth - 1, alpha, beta, false);
                gamestate.undo_move(col);

                if eval > max_eval {
                    max_eval = eval as isize;
                    best_col = good_col;
                }
                alpha = alpha.max(eval as isize);
                if beta <= alpha {
                    break;
                }
            }
            return (max_eval, best_col)
        } else {
            let mut min_eval = isize::MAX;
            for &col in &valid_moves {
                gamestate.play_move(col, false);
                let (eval, good_col) = self.minimax(gamestate, depth - 1, alpha, beta, true);
                //because of recursion, this is somewhere gone on the call stack. 
                gamestate.undo_move(col);
                
                if eval < min_eval {
                    min_eval = eval as isize;
                    best_col = good_col;
                }
                beta = beta.min(eval as isize);
                if beta <= alpha {
                    break;
                }
            }
                return (min_eval, best_col)
            }
        }
    }

impl Player for MinMaxPlayer {
    //assign weights to each position on the board.
    //algorithm will explore depth of 3.
    fn make_move(&mut self, gamestate: &mut GameState) {
        //both players start with worst possible score. Alpha is -infinity, beta is +infinity
        //whenever maximum score of minimizing player crosses minimum score of maximizing player
        //branches of the tree are 'pruned'
        let (_score, col) = self.minimax(gamestate, 3, isize::MIN, isize::MAX, true);
        gamestate.play_move(col, false);

    }
}
