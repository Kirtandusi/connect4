use std::io;
use rand::Rng;
extern crate term;
fn main() {
    input();
}

/*
 *7 columns by 6 rows. The colors red and black are available. They will be displayed using X, Y respectively. Unused squares will be 0. Input rows 0-6. 
 */
fn input() {
    println!("Welcome to Connect 4. Player goes first. Player moves get marked with a 1, and CPU moves get marked with a 2.");
    let mut loss = false;
    //vector of vectors. 
    let mut board: Vec<Vec<usize>> = vec![vec![0; 7]; 6];
    board_to_string(&mut board);
    let mut input_str: String = String::new();
    while !loss & is_not_full(&mut board) { //input cycle
        input_str.clear();
        println!("Please choose a column to drop your piece. Input range is 1-7");
        io::stdin().read_line(&mut input_str).expect("Failed to read input");
       
        let mut column: usize = match input_str.trim().parse() {
            Ok(num) => num,
            Err(_) => {
                println!("Invalid input, please try again");
                continue;
            }
        };
        
        if column < 1 || column > 7 {
            println!("Invalid input, please try again");
            continue;
        }
        column = column - 1;
        println!();
        //must pass address for ownership reasons
        if !check_if_full(&mut board, column) {
            play_move(column, &mut board, true);
        }
        
        if check_for_win(&mut board) {
            println!("You win!");
            loss = true;
        }
       
        board_to_string(&mut board);
        println!("This is the new board after your move.");
        cpu_random(&mut board, &mut vec![]);
            
        println!();
        board_to_string(&mut board);
        if check_for_win(&mut board) {
            println!("CPU wins!");
            loss = true;
        } else {
            print!("This is the new board after the CPU's turn. ");
            if is_not_full(&mut board) {
                println!("Your move.");
            }
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
fn top(_board: &mut Vec<Vec<usize>>, column: usize) -> usize {
    //returns the topmost row of a column
    for i in (0..6).rev() {
        if _board[i][column] == 0 {
            return i;
        }
    }
    return 6;
}


/*
* True = player move. False = CPU move.
*/
fn play_move(mut _column: usize, _board: &mut Vec<Vec<usize>>, side: bool) {
    //precipitate.
    let top_of_column: usize = top(_board, _column);
    if top_of_column == 6 {
        println!("Column is full, please choose another column");
    }
    if side { //player move. This is signified by X
        _board[top_of_column][_column] = 1;
    } else {
       _board[top_of_column][_column] = 2;
    }
    return;
}


fn cpu_random(board: &mut Vec<Vec<usize>>, exclude: &mut Vec<usize>) {
    if exclude.len() == 7 {
        println!("Board is full, game is a tie!");
        return;
    }
    let mut rng = rand::thread_rng();
    let valid_columns: Vec<usize> = (0..7).filter(|&x| !exclude.contains(&x)).collect();
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
    return;
}
fn check_if_full(board: &mut Vec<Vec<usize>>, column: usize) -> bool {
    return board[0][column] != 0;

}
fn board_to_string(board: &mut Vec<Vec<usize>>) {
    let mut terminal = term::stdout().unwrap();
    for i in 0..6{
        for j in 0..7{
            if board[i][j] == 1 {
                terminal.fg(term::color::YELLOW).unwrap();
            } else if board[i][j] == 2 {
                terminal.fg(term::color::RED).unwrap();
            } else {
                terminal.fg(term::color::WHITE).unwrap();
            }
            print!("{} ", board[i][j]);
        }
        terminal.fg(term::color::WHITE).unwrap();
        println!();
    }
}

fn is_not_full(board: &mut Vec<Vec<usize>>) -> bool {
    for i in 0..7 {
        for j in 0..6 {
            if board[j][i] == 0 {
                return true;
            }
        }
    }
    return false;

}
// fn cpu_player(board: &mut Box<[usize]>) {
//     //input neural network things here
//     return;
// }


