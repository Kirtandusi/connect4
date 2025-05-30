use connect4::game_state::GameState;
use connect4::player::Player;
use connect4::random_player::RandomPlayer;
use connect4::minimax_player::MinMaxPlayer;
use connect4::neuralnetwork::NeuralNetPlayer;

fn simulate_n_games(mut player1: Box<dyn Player>, mut player2: Box<dyn Player>, number_of_tests: usize) -> (usize, usize, usize) {
    let mut player1_wins = 0;
    let mut player2_wins = 0;
    let mut draws = 0;

    for _ in 0..number_of_tests {


        let mut board = GameState::new();
        let mut winner = None;

        while board.is_not_full() {
            player1.make_move(&mut board);
            if board.check_for_win() {
                winner = Some(player1.get_name());
                break;
            }

            if !board.is_not_full() {
                break;
            }

            player2.make_move(&mut board);
            if board.check_for_win() {
                winner = Some(player2.get_name());
                break;
            }
        }

        match winner {
            Some(name) if name == player1.get_name() => player1_wins += 1,
            Some(name) if name == player2.get_name() => player2_wins += 1,
            _ => draws += 1,
        }
    }

    (player1_wins, player2_wins, draws)
}

#[test]
fn test_minimax_vs_random() {
    let player1 = Box::new(MinMaxPlayer::new(true)) as Box<dyn Player>;
    let player2 = Box::new(RandomPlayer::new(false)) as Box<dyn Player>;
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


#[test]
fn test_random_vs_neuralnet() {
    let mut nn = NeuralNetPlayer::new(true);
    nn.train_generalized(100000);
    let player1 = Box::new(nn) as Box<dyn Player>;
    let player2 = Box::new(RandomPlayer::new(false)) as Box<dyn Player>;

    let num_games = 50;
    let (neuralnet_wins, random_wins, draws) = simulate_n_games(player1, player2, num_games);

    println!("Simulated {} games:", num_games);
    println!("Neural Net wins: {}", neuralnet_wins);
    println!("Random wins: {}", random_wins);
    println!("Draws: {}", draws);

    assert!(neuralnet_wins > random_wins, "Expected Neural Net to outperform Random.");
    assert_eq!(neuralnet_wins + random_wins + draws, num_games);
}
#[test]
fn test_minimax_vs_neuralnet() {
    let mut nn = NeuralNetPlayer::new(true);
    nn.train_generalized(30000);
    let player1 = Box::new(nn) as Box<dyn Player>;
    let player2 = Box::new(MinMaxPlayer::new(false)) as Box<dyn Player>;
    let num_games = 50;
    let (neuralnet_wins, minimax_wins, draws) = simulate_n_games(player1, player2, num_games);
    println!("Simulated {} games:", num_games);
    println!("Neural Net wins: {}", neuralnet_wins);
    println!("Minimax wins: {}", minimax_wins);
    println!("Draws: {}", draws);

    assert!(neuralnet_wins > minimax_wins, "Expected Neural Net to outperform Minimax.");
    assert_eq!(neuralnet_wins + minimax_wins + draws, num_games);
}


