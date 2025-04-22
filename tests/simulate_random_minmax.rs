use connect4::game_state::GameState;
use connect4::player::Player;
use connect4::random_player::RandomPlayer;
use connect4::minimax_player::MinMaxPlayer;
/// Simulates `number_of_tests` games and asserts basic stats.
fn simulate_n_games(mut player1: Box<dyn Player>, mut player2: Box<dyn Player>, number_of_tests: usize) -> (usize, usize, usize) {
    let mut random_wins = 0;
    let mut minimax_wins = 0;
    let mut draws = 0;

    for _ in 0..number_of_tests {


        let mut board = GameState::new();
        let mut winner = None;

        while board.is_not_full() {
            player1.make_move(&mut board);
            if board.check_for_win() {
                winner = Some("minimax");
                break;
            }

            if !board.is_not_full() {
                break;
            }

            player2.make_move(&mut board);
            if board.check_for_win() {
                winner = Some("random");
                break;
            }
        }

        match winner {
            Some("minimax") => minimax_wins += 1,
            Some("random") => random_wins += 1,
            _ => draws += 1,
        }
    }

    (minimax_wins, random_wins, draws)
}

#[test]
fn test_minimax_vs_random() {
    let mut player1 = Box::new(MinMaxPlayer::new(true)) as Box<dyn Player>;
    let mut player2 = Box::new(RandomPlayer::new(false)) as Box<dyn Player>;
    let num_games = 50;
    let (minimax_wins, random_wins, draws) = simulate_n_games(player1, player2, num_games);

    println!("Simulated {} games:", num_games);
    println!("Minimax wins: {}", minimax_wins);
    println!("Random wins: {}", random_wins);
    println!("Draws: {}", draws);

    // Soft check: we expect minimax to win more often
    assert!(minimax_wins > random_wins, "Expected Minimax to outperform Random.");
    assert_eq!(minimax_wins + random_wins + draws, num_games);
}
//next test needs to be to test neural net vs minimax!
