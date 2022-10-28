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

    def __init__(self, chessboard_size, color, time_out):
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
        self.weighted_map = np.array([[-500, 25, -10, -5, -5, -10, 25, -500],
                                      [25, 45, -1, -1, -1, -1, 45, 25],
                                      [-10, -1, -3, -2, -2, -3, -1, -10],
                                      [-5, -1, -2, -1, -1, -2, -1, -5],
                                      [-5, -1, -2, -1, -1, -2, -1, -5],
                                      [-10, -1, -3, -2, -2, -3, -1, -10],
                                      [25, 45, -1, -1, -1, -1, 45, 25],
                                      [-500, 25, -10, -5, -5, -10, 25, -500]])

    def go(self, chessboard):
        self.candidate_list.clear()
        self.next_state.clear()
        self.next_state = self.get_state(chessboard, self.color)
        self.candidate_list = list(self.next_state.keys())
        if len(self.candidate_list) == 0:
            return self.candidate_list
        self.candidate_list.append(random.choice(self.candidate_list))
        count = self.count_chess(chessboard)
        if count < 50:
            self.search(chessboard, self.color, d=2)
        else:
            self.search(chessboard, self.color, d=4)
        return self.candidate_list

    def search(self, chessboard, color, d=2):
        flag = 0
        last = 4.9
        while 1:
            if flag > 3:
                break
            s = time.time()
            pos = self.alpha_beta(chessboard, color, d)
            if pos is None:
                break
            if pos not in self.precedence[4]:
                self.candidate_list.append(pos)
            e = time.time()
            usage = e - s
            if 2 * usage < last:
                last -= usage
                last -= 0.1
                d += 1
                flag += 1
            else:
                break

    def alpha_beta(self, chessboard, color, d=4):
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
        def is_frontier(board, x, y):
            for dx, dy in [(dx, dy) for dx in range(-1, 2) for dy in range(-1, 2) if dx != 0 or dy != 0]:
                if self.judge(x + dx, y + dy) and board[x + dx][y + dy] != 0:
                    return True
            return False

        def judge_stable(board, current_color, x, y, z):
            direction_x = [-1, 1, 0, 0, -1, 1, 1, -1]
            direction_y = [0, 0, -1, 1, -1, 1, -1, 1]
            bound_x = x + direction_x[z]
            bound_y = y + direction_y[z]
            while self.judge(bound_x, bound_y) and board[bound_x][bound_y] == current_color:
                bound_x += direction_x[z]
                bound_y += direction_y[z]
            if not self.judge(bound_x, bound_y) or \
                    (self.judge(bound_x, bound_y) and board[bound_x][bound_y] == COLOR_NONE):
                return True
            return False

        def calculate(board, current_color, x, y):
            frontier = 0
            stability = 0
            if is_frontier(board, x, y):
                frontier += 1
            for z in [0, 2, 4, 6]:
                flag1 = judge_stable(board, current_color, x, y, z)
                flag2 = judge_stable(board, current_color, x, y, z + 1)
                if flag1 and flag2:
                    stability += 1
            return frontier, stability

        def weighting(weighted_map):
            return np.sum(chessboard * weighted_map * self.color)

        cnt = self.count_chess(chessboard)
        my_frontier = 0
        my_stability = 0
        op_frontier = 0
        op_stability = 0
        for i in range(8):
            for j in range(8):
                if chessboard[i][j] == self.color:
                    result = calculate(chessboard, self.color, i, j)
                    my_frontier += result[0]
                    my_stability += result[1]
                elif chessboard[i][j] == -self.color:
                    result = calculate(chessboard, -self.color, i, j)
                    op_frontier += result[0]
                    op_stability += result[1]
        if cnt < 30:
            return weighting(weighted_map=self.weighted_map)
        elif cnt < 45:
            return weighting(weighted_map=self.weighted_map) + 6 * (my_frontier - op_frontier) + \
                   8 * (op_stability - my_stability)
        elif cnt < 58:
            return weighting(weighted_map=self.weighted_map) + 16 * (op_stability - my_stability)
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
