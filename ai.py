import time
import random
from math import inf
from numba import jit
from functools import cmp_to_key

import numpy as np

random.seed(0)

COLOR_BLACK = -1
COLOR_WHITE = 1
COLOR_NONE = 0

precedence0 = [(0, 1), (0, 6), (1, 0), (1, 1), (1, 6), (1, 7), (6, 0), (6, 1), (6, 6), (6, 7), (7, 1), (7, 6)]
precedence1 = [(1, 2), (1, 3), (1, 4), (1, 5), (2, 1), (3, 1), (4, 1), (5, 1), (2, 6), (3, 6), (4, 6), (5, 6),
               (6, 2), (6, 3), (6, 4), (6, 5)]
precedence2 = [(0, 2), (0, 3), (0, 4), (0, 5), (2, 0), (3, 0), (4, 0), (5, 0), (2, 7), (3, 7), (4, 7), (5, 7),
               (7, 2), (7, 3), (7, 4), (7, 5)]
precedence3 = [(2, 2), (2, 3), (2, 4), (2, 5), (3, 2), (3, 5), (4, 2), (4, 5), (5, 2), (5, 3), (5, 4), (5, 5)]
precedence4 = [(0, 0), (0, 7), (7, 0), (7, 7)]


def tuple_to_int(t: tuple):
    if t in precedence0:
        return 0
    elif t in precedence1:
        return 1
    elif t in precedence2:
        return 2
    elif t in precedence3:
        return 3
    elif t in precedence4:
        return 4
    else:
        return 3


def cmp(t1: tuple, t2: tuple):
    num1 = tuple_to_int(t1)
    num2 = tuple_to_int(t2)
    return num1 - num2


class AI(object):

    def __init__(self, chessboard_size, color, time_out, data):
        self.data = data
        self.chessboard_size = chessboard_size
        self.color = color
        self.time_out = time_out
        self.candidate_list = []
        self.next_state = {}
        self.precedence = (precedence0, precedence1, precedence2, precedence3, precedence4)
        self.cache = {}
        self.inner_squares = [(3, 3), (4, 4), (4, 3), (3, 4)]
        self.x_squares = [(1, 1), (1, 6), (6, 1), (6, 6)]
        self.c_squares = [(0, 1), (0, 6), (1, 0), (1, 7), (6, 0), (6, 7), (7, 1), (7, 6)]
        self.corner_squares = [(0, 0), (0, 7), (7, 0), (7, 7)]
        self.side_squares = [(x, y) for x in range(8) for y in range(8) if (x == 0 or x == 7 or y == 0 or y == 7)]
        self.where = np.where
        self.sum = np.sum
        self.inf = inf
        self.weighted_map0 = np.array([[data[0], data[1], data[2], data[5], data[5], data[2], data[1], data[0]],
                                       [data[1], data[3], data[4], data[9], data[9], data[4], data[3], data[1]],
                                       [data[2], data[4], data[6], data[7], data[7], data[6], data[4], data[2]],
                                       [data[5], data[9], data[7], data[8], data[8], data[7], data[9], data[5]],
                                       [data[5], data[9], data[7], data[8], data[8], data[7], data[9], data[5]],
                                       [data[2], data[4], data[6], data[7], data[7], data[6], data[4], data[2]],
                                       [data[1], data[3], data[4], data[9], data[9], data[4], data[3], data[1]],
                                       [data[0], data[1], data[2], data[5], data[5], data[2], data[1], data[0]]]
                                      )
        self.weighted_map1 = np.array([[data[10], data[11], data[12], data[15], data[15], data[12], data[11], data[10]],
                                       [data[11], data[13], data[14], data[19], data[19], data[14], data[13], data[11]],
                                       [data[12], data[14], data[16], data[17], data[17], data[16], data[14], data[12]],
                                       [data[15], data[19], data[17], data[18], data[18], data[17], data[19], data[15]],
                                       [data[15], data[19], data[17], data[18], data[18], data[17], data[19], data[15]],
                                       [data[12], data[14], data[16], data[17], data[17], data[16], data[14], data[12]],
                                       [data[11], data[13], data[14], data[19], data[19], data[14], data[13], data[11]],
                                       [data[10], data[11], data[12], data[15], data[15], data[12], data[11], data[10]]]
                                      )
        self.weighted_map2 = np.array([[data[20], data[21], data[22], data[25], data[25], data[22], data[21], data[20]],
                                       [data[21], data[23], data[24], data[29], data[29], data[24], data[23], data[21]],
                                       [data[22], data[24], data[26], data[27], data[27], data[26], data[24], data[22]],
                                       [data[25], data[29], data[27], data[28], data[28], data[27], data[29], data[25]],
                                       [data[25], data[29], data[27], data[28], data[28], data[27], data[29], data[25]],
                                       [data[22], data[24], data[26], data[27], data[27], data[26], data[24], data[22]],
                                       [data[21], data[23], data[24], data[29], data[29], data[24], data[23], data[21]],
                                       [data[20], data[21], data[22], data[25], data[25], data[22], data[21], data[20]]]
                                      )

    def go(self, chessboard):
        self.candidate_list.clear()
        self.next_state.clear()
        self.next_state = self.get_state(chessboard, self.color)
        self.candidate_list = list(self.next_state.keys())
        if len(self.candidate_list) == 0:
            return self.candidate_list
        self.candidate_list.append(random.choice(self.candidate_list))
        count = self.count_chess(chessboard)
        if count < 48:
            self.search(chessboard, self.color, d=2)
        else:
            self.search(chessboard, self.color, d=2)
        return self.candidate_list

    def search(self, chessboard, color, d=2):
        # flag = 0
        # last = 4.88
        # time_used = 0
        while 1:
            # s = time.time()
            pos = self.alpha_beta(chessboard, color, d)
            if pos is None:
                break
            self.candidate_list.append(pos)
            break
            # e = time.time()
            # if flag > 2:
            #     break
            # usage = e - s
            # time_used += usage
            # if 2 * usage < last:
            #     last -= usage
            #     last -= 0.1
            #     d += 1
            #     flag += 1
            # else:
            #     break

    def alpha_beta(self, chessboard, color, d=1):
        def max_value(board, current_color, alpha, beta, depth):
            state = self.get_state(board, current_color)
            if len(state) == 0:
                return self.calculate_score(board, self.color)
            if depth >= d:
                return self.evaluate(board)
            val = -self.inf
            sorted_state = sorted(state, key=cmp_to_key(cmp))
            get = state.get
            for pos in sorted_state:
                sub_board = get(pos)
                val = max(val, min_value(sub_board, -current_color, alpha, beta, depth + 1))
                if val >= beta:
                    return val
                alpha = max(alpha, val)
            return val

        def min_value(board, current_color, alpha, beta, depth):
            state = self.get_state(board, current_color)
            if len(state) == 0:
                return self.calculate_score(board, self.color)
            if depth >= d:
                return self.evaluate(board)
            val = self.inf
            sorted_state = sorted(state, key=cmp_to_key(cmp), reverse=True)
            get = state.get
            for pos in sorted_state:
                sub_board = get(pos)
                val = min(val, max_value(sub_board, -current_color, alpha, beta, depth + 1))
                if val <= alpha:
                    return val
                beta = min(beta, val)
            return val

        best_score = -self.inf
        temp = self.inf
        best_action = None
        next_state = self.get_state(chessboard, color)
        next_sorted_state = sorted(next_state, key=cmp_to_key(cmp), reverse=True)
        get_board = next_state.get
        for move in next_sorted_state:
            next_board = get_board(move)
            v = min_value(board=next_board, current_color=-color, alpha=best_score, beta=temp, depth=1)
            if v > best_score:
                best_score = v
                best_action = move
        return best_action

    def evaluate(self, chessboard):
        def weighting(weighted_map):
            return np.sum(chessboard * weighted_map * self.color)

        cnt = self.count_chess(chessboard)
        my_num = np.sum(chessboard == self.color)
        op_num = np.sum(chessboard == -self.color)
        if my_num == 0:
            return 50000
        if op_num == 0:
            return -50000
        if cnt < 20:
            return weighting(weighted_map=self.weighted_map0)
        elif cnt < 36:
            return (op_num - my_num) * self.data[30] * self.data[30] + weighting(weighted_map=self.weighted_map1) * (
                        1 - self.data[32])
        elif cnt < 50:
            return (op_num - my_num) * self.data[31] * self.data[31] + weighting(weighted_map=self.weighted_map1) * (
                        1 - self.data[33])
        else:
            return self.calculate_score(chessboard, self.color)

    def judge(self, x, y):
        return 0 <= x < self.chessboard_size and 0 <= y < self.chessboard_size

    def get_state(self, chessboard, color):

        def get_valid_pos(x, y, dx, dy):
            pos_set = []
            including = False
            append = pos_set.append
            judge = self.judge
            while 1:
                x += dx
                y += dy
                if not judge(x, y) or chessboard[x][y] == COLOR_NONE:
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
        indexes = self.where(chessboard == COLOR_NONE)
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

    @staticmethod
    def get_key(chessboard, color):
        board_map = chessboard.copy()
        board_map[board_map == color] = 1
        board_map[board_map == -color] = -1
        board = tuple([tuple(e) for e in board_map])
        key = tuple([board, color])
        return key

    @staticmethod
    def calculate_score(chessboard, color):
        black_num = np.sum(chessboard == COLOR_BLACK)
        white_num = np.sum(chessboard == COLOR_WHITE)
        return (white_num - black_num) * (-color) * 15000

    @staticmethod
    @jit(nopython=True)
    def count_chess(chessboard):
        return np.sum(chessboard != 0)
