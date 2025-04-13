use crate::game_state::GameState;
use crate::player::Player;
use rand::Rng;
pub struct RandomPlayer {
    player: bool,
}
impl RandomPlayer {
    pub(crate) fn new(player: bool) -> Self {
        RandomPlayer { player }
    }
}

impl Player for RandomPlayer {
    fn make_move(&mut self, gamestate: &mut GameState) {
        if gamestate.exclude.len() == 7 {
            println!("Board is full, game is a tie!");
            return;
        }
        let mut rng = rand::thread_rng();
        let valid_columns: Vec<usize> = (0..7)
            .filter(|&x| !gamestate.exclude.contains(&x))
            .collect();
        if valid_columns.is_empty() {
            println!("No valid columns left, game over!");
            return;
        }
        let column = valid_columns[rng.gen_range(0..valid_columns.len())];
        if !gamestate.check_if_full(column) {
            gamestate.play_move(column, self.player); // CPU move
        } else {
            gamestate.exclude.push(column);
            self.make_move(gamestate)
        }
    }

    fn get_name(&self) -> &str {
       "Random Player"
    }
}
