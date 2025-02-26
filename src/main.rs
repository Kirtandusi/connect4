use std::io;
use rand::Rng;
fn main() {
    input();
}

/*
 *7 columns by 6 rows. The colors red and black are available. They will be displayed using X, Y respectively. Unused squares will be 0. Input rows 0-6. 
 */
fn input() {
    println!("Welcome to Connect 4. Player goes first.");
    let mut loss = false;
    //vector of vectors. 
    let mut board: Vec<Vec<usize>> = vec![vec![0; 7]; 6];
    board_to_string(&mut board);
    let mut input_str: String = String::new();
    while !loss { //input cycle
        println!("Please choose a column to drop your piece. Input range is 0-7");
        io::stdin().read_line(&mut input_str).expect("Failed to read input");
       
        let column: usize = match input_str.trim().parse() {
            Ok(num) => num,
            Err(_) => {
                println!("Invalid input, please enter a number");
                continue;
            }
        };

        println!();
        //must pass address for ownership reasons
        play_move(column, &mut board, true);
        if check_for_win(&mut board) {
            println!("You win!");
            loss = true;
        }
       
        board_to_string(&mut board);
        println!("This is the new board after your move.");
        cpu_random(&mut board, vec![]);
        println!();
        board_to_string(&mut board);
        if check_for_win(&mut board) {
            println!("CPU wins!");
            loss = true;
        } else {
            println!("This is the new board after the CPU's turn. Your move.");
            println!();
        }
    }
    println!("Game over");
    return;
}
fn check_for_win(_board: &mut Vec<Vec<usize>>) -> bool {
    //check which side wins. 
    return false;
}
/*
* True = player move. False = CPU move.
*/
fn play_move(_column: usize, _board: &mut Vec<Vec<usize>>, side: bool) {
    //precipitate.
    if side { //player move. This is signified by X
        _board[0][_column] = 1;
    } else {
        _board[0][_column] = 2;
    }
    return;
}
fn cpu_random(board: &mut Vec<Vec<usize>>, exclude: &mut Vec<usize>) {
    if (exclude.len() == 7) {
        println!("CPU cannot play, game over");
        return;
    }
    let mut rng = rand::thread_rng();
    let mut valid_columns: Vec<usize> = (0..7).filter(|&x| !exclude.contains(&x)).collect();
    if valid_columns.is_empty() {
        println!("No valid columns left, game over!");
        return;
    }
    let column = valid_columns[rng.gen_range(0..valid_columns.len())];
    if !check_if_full(board, column) {
        play_move(column, board, false);
    } else {
        exclude.push(column);
        cpu_random(board, exclude);
    }
}
fn check_if_full(board: &mut Vec<Vec<usize>>, column: usize) -> bool {
    return board[0][column] != 0;

}
fn board_to_string(board: &mut Vec<Vec<usize>>) {
    for i in 0..6{
        for j in 0..7{
            print!("{} ", board[i][j]);
        }
        println!();
    }
}
// fn cpu_player(board: &mut Box<[i32]>) {
//     //input neural network things here
//     return;
// }
