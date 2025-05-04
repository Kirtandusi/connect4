use crate::game_state::GameState;

/*
   This Player class is used as an interface.
*/
pub trait Player {
    fn make_move(&mut self, game_state: &mut GameState);
    #[allow(dead_code)]
    fn get_name(&self) -> &str;
}
