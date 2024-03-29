import random

import numpy as np

COLOR_BLACK = -1
COLOR_WHITE = 1
COLOR_NONE = 0

random.seed(0)


class Idiot(object):

    def __init__(self, chessboard_size, color, time_out):
        self.chessboard_size = chessboard_size
        self.color = color
        self.time_out = time_out

        self.chessboard = None
        self.candidate_list = []
        self.next_state = {}

    def go(self, chessboard):
        self.candidate_list.clear()
        self.next_state.clear()
        self.chessboard = chessboard
        self.next_state = self.get_state(chessboard, self.color)
        self.candidate_list = list(self.next_state.keys())
        if len(self.candidate_list) == 0:
            return self.candidate_list
        self.candidate_list.append(random.choice(self.candidate_list))
        return self.candidate_list

    def judge(self, x: int, y: int) -> bool:
        return 0 <= x < self.chessboard_size and 0 <= y < self.chessboard_size

    def get_state(self, chessboard, color):

        def get_valid_pos(x, y, dx, dy):
            pos_set = []
            including = False
            append = pos_set.append
            while 1:
                x += dx
                y += dy
                if not self.judge(x, y) or chessboard[x][y] == COLOR_NONE:
                    break
                elif chessboard[x][y] == color and not including:
                    break
                elif chessboard[x][y] == -color:
                    append((x, y))
                    including = True
                elif chessboard[x][y] == color and including:
                    return pos_set
            return None

        def test_all_directions(x, y):
            pos_set = []
            directions = [(dx, dy) for dx in range(-1, 2) for dy in range(-1, 2) if dx != 0 or dy != 0]
            for dx, dy in directions:
                subset = get_valid_pos(x, y, dx, dy)
                if subset is not None:
                    pos_set += subset
            return (x, y), pos_set

        next_state = {}
        indexes = np.where(chessboard == COLOR_NONE)
        indexes = tuple(zip(indexes[0], indexes[1]))
        for idx in indexes:
            idx, reversed_color_set = test_all_directions(idx[0], idx[1])
            if len(reversed_color_set) > 0:
                new_chessboard = chessboard.copy()
                for pos in reversed_color_set:
                    new_chessboard[pos[0]][pos[1]] = color
                new_chessboard[idx[0]][idx[1]] = color
                next_state[idx] = new_chessboard
        return next_state