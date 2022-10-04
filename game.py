import time

import numpy as np

from main import *

ai1 = AI(8, COLOR_BLACK, 5)
ai2 = AI(8, COLOR_WHITE, 5)
board = np.zeros((8, 8))
board[3][3] = COLOR_WHITE
board[4][4] = COLOR_WHITE
board[3][4] = COLOR_BLACK
board[4][3] = COLOR_BLACK

s = time.time()
i = 1

while True:
    print(i)
    i += 1
    l1 = ai1.go(board)
    if len(l1) != 0:
        piece1 = l1[-1]
        board = ai1.next_state[piece1]
    l2 = ai2.go(board)
    if len(l2) != 0:
        piece2 = l2[-1]
        board = ai2.next_state[piece2]

    if time.time() - s > 200:
        break

print(board)
print(len(tuple(zip(np.where(board == 1)[0]))))
print(len(tuple(zip(np.where(board == -1)[0]))))
