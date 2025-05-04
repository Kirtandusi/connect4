use crate::game_state::GameState;
use rand::Rng;

/*
   helper file used for back propagation.
*/
pub struct Connect4Env {
    game: GameState,
    player: bool,
}

impl Connect4Env {
    pub fn new(player: bool) -> Self {
        Self {
            game: GameState::new(),
            player,
        }
    }

    pub fn reset(&mut self) -> Vec<f64> {
        self.game = GameState::new();
        self.game.to_input_vector()
    }

    //change rewards?
    pub fn step(&mut self, action: usize) -> (Vec<f64>, f64, bool) {
        let mut reward = 0.0;
        self.game.play_move(action, self.player);

        if self.game.check_for_win() {
            reward = 1.0;
            return (self.game.to_input_vector(), reward, true);
        }

        if !self.game.is_not_full() {
            // Game is a draw
            reward = 0.5;
            return (self.game.to_input_vector(), reward, true);
        }

        // Let opponent (random) play
        let opponent_action = self.sample_random_action();
        self.game.play_move(opponent_action, !self.player);

        if self.game.check_for_win() {
            reward = -1.0;
            return (self.game.to_input_vector(), reward, true);
        }

        if !self.game.is_not_full() {
            reward = 0.5;
            return (self.game.to_input_vector(), reward, true);
        }

        (self.game.to_input_vector(), reward, false)
    }

    pub fn valid_moves(&self) -> Vec<usize> {
        self.game.get_valid_moves()
    }

    pub fn sample_random_action(&self) -> usize {
        let valid = self.valid_moves();
        let mut rng = rand::rng();
        let idx = rng.random_range(0..valid.len());
        valid[idx]
    }

    pub fn get_state_vector(&self) -> Vec<f64> {
        self.game.to_input_vector()
    }
}
