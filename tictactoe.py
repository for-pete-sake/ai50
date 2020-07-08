"""
Tic Tac Toe Player
"""

import math
import copy
from copy import deepcopy
from random import randint

X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    # start counting the number of X and O
    number_of_X = 0
    number_of_O = 0

    for row in board:
        for cell in row:
            if cell == X:
                number_of_X += 1
            elif cell == O:
                number_of_O +=1
            

    # if X goes first, number of X > number of O, then it's O's turn
    # if then O goes after, number of O == number of X and if not terminal state,
    # then it's X's turn

    if number_of_X <= number_of_O:
        return X
    #elif number_of_O == number_of_X and not terminal(board):
       # return X
    else:
        #return None
        return O

def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    # create a set of all possible moves
    possible_moves = set()

    # all the possible move in 3x3 board
    for i in range(3):
        for j in range(3):
            # if cell empty, then can make a move at that cell
            if board[i][j] == EMPTY:
                possible_moves.add((i, j))
    # else, return actions
    return possible_moves


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    board_copy = deepcopy(board)

    if board_copy[action[0]][action[1]] != EMPTY:
        raise Exception('Place of action must be Empty')
    else:
        board_copy[action[0]][action[1]] = player(board)

    return board_copy



def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    # if 3 cells in a row horizontally, vertically or diagonally and not None -> win
    # else, tie -> return None
    for i in range(3):
        if (board[i][0] == board[i][1]== board[i][2]) and (board[i][0] != EMPTY):
            return board[i][0]
        if (board[0][i] == board[1][i]== board[2][i]) and (board[0][i] != EMPTY):
            return board[0][i]
        
        if (board[0][0] == board[1][1]== board[2][2]) or (board[0][2] == board[1][1]== board[2][0])\
                and (board[1][1] != EMPTY):
            return board[1][1]

        return None
        



def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    if winner(board) is not None:
        return True

    for i in range(3):
        for j in range(3):
            if board[i][j] == EMPTY:
                return False

    return True

def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    if winner(board) == X:
        return 1
    elif winner(board) == O:
        return -1
    else:
        return 0


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    # output: action (i, j)
    # track each move
    #player_name = player(board)

    if terminal(board):
        return None

    if player(board) == X:
        score = -math.inf
        best_action = None

        for action in actions(board):
            min_val = MIN_VALUE(result(board, action))

            if min_val > score:
                score = min_val
                best_action = action

        return best_action

    elif player(board) == O:
        score = math.inf
        best_action = None

        for action in actions(board):
            max_val = MAX_VALUE(result(board, action))

            if max_val < score:
                score = max_val
                best_action = action
   
        return best_action



def MAX_VALUE(board):
    if terminal(board):
        return utility(board)

    num = -math.inf

    for action in actions(board):
        num = max(num, MIN_VALUE(result(board, action)))
    return num

def MIN_VALUE(board):
    if terminal(board):
        return utility(board)

    num = math.inf

    for action in actions(board):
        num = min(num, MAX_VALUE(result(board, action)))
    return num

