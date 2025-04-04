use crate::game_state::GameState;
pub trait Player {
    fn make_move(&mut self, game_state: &mut GameState);
    fn get_name(&self) -> &str;
}
