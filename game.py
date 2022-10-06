from main import *

BLACK_PIECE = "⚫"
WHITE_PIECE = "○"
NONE_PIECE = "✖"


def get_winner(chessboard):
    black = len(tuple(zip(np.where(chessboard == COLOR_BLACK)[0])))
    white = len(tuple(zip(np.where(chessboard == COLOR_WHITE)[0])))
    print("Black: {}".format(black))
    print("White: {}".format(white))
    if black > white:
        return "White Win"
    elif black < white:
        return "Black Win"
    else:
        return "Draw"


def show(chessboard):
    for i in range(8):
        for j in range(8):
            if chessboard[i][j] == COLOR_WHITE:
                print(WHITE_PIECE, end=" ")
            elif chessboard[i][j] == COLOR_BLACK:
                print(BLACK_PIECE, end=" ")
            else:
                print(NONE_PIECE, end=" ")
        print()


def judge(chessboard):
    if len(tuple(zip(np.where(chessboard != 0)[0]))) == 64:
        return True
    return False


flag = False
ai1 = AI(8, COLOR_BLACK, 5)
ai2 = AI(8, COLOR_WHITE, 5)
board = np.zeros((8, 8))
board[3][3] = COLOR_WHITE
board[4][4] = COLOR_WHITE
board[3][4] = COLOR_BLACK
board[4][3] = COLOR_BLACK

while True:
    result = ai1.go(board)
    if len(result) > 0:
        piece = result[-1]
        board = ai1.next_state[piece]
        if flag:
            flag = False
    else:
        if flag:
            break
        flag = True
    show(board)
    time.sleep(1)

    if judge(board):
        break

    result = ai2.go(board)
    if len(result) > 0:
        piece = result[-1]
        board = ai2.next_state[piece]
        if flag:
            flag = False
    else:
        if flag:
            break
        flag = True
    show(board)
    time.sleep(1)

    if judge(board):
        break

print(get_winner(board))
