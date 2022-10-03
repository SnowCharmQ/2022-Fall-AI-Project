from main import *

ai1 = AI(8, COLOR_BLACK, 5)
ai2 = AI(8, COLOR_WHITE, 5)
board = np.zeros((8, 8))
board[3][3] = COLOR_WHITE
board[4][4] = COLOR_WHITE
board[3][4] = COLOR_BLACK
board[4][3] = COLOR_BLACK

while True:
    l1 = ai1.go(board)
    piece1 = l1[-1]
    board = ai1.next_state[piece1]
    print(board)
    l2 = ai2.go(board)
    piece2 = l2[-1]
    board = ai2.next_state[piece2]
    print(board)
    break
