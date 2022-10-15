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
        self.beginning = True
        self.inner_squares = [(3, 3), (4, 4), (4, 3), (3, 4)]
        self.x_squares = [(1, 1), (1, 6), (6, 1), (6, 6)]
        self.c_squares = [(0, 1), (0, 6), (1, 0), (1, 7), (6, 0), (6, 7), (7, 1), (7, 6)]
        self.corner_squares = [(0, 0), (0, 7), (7, 0), (7, 7)]
        self.side_squares = [(x, y) for x in range(8) for y in range(8) if (x == 0 or x == 7 or y == 0 or y == 7)]
        self.where = np.where
        self.sum = np.sum
        self.inf = inf
        self.weighted_map0 = np.array([[-500, 25, -10, -5, -5, -10, 25, -500],
                                      [25, 45, -1, -1, -1, -1, 45, 25],
                                      [-10, -1, -3, -2, -2, -3, -1, -10],
                                      [-5, -1, -2, -1, -1, -2, -1, -5],
                                      [-5, -1, -2, -1, -1, -2, -1, -5],
                                      [-10, -1, -3, -2, -2, -3, -1, -10],
                                      [25, 45, -1, -1, -1, -1, 45, 25],
                                      [-500, 25, -10, -5, -5, -10, 25, -500]])
        self.weighted_map1 = np.array([[-500, 25, -10, -5, -5, -10, 25, -500],
                                      [25, 45, -1, -1, -1, -1, 45, 25],
                                      [-10, -1, -3, -2, -2, -3, -1, -10],
                                      [-5, -1, -2, -1, -1, -2, -1, -5],
                                      [-5, -1, -2, -1, -1, -2, -1, -5],
                                      [-10, -1, -3, -2, -2, -3, -1, -10],
                                      [25, 45, -1, -1, -1, -1, 45, 25],
                                      [-500, 25, -10, -5, -5, -10, 25, -500]])
        self.weighted_map2 = np.array([[-500, 25, -10, -5, -5, -10, 25, -500],
                                      [25, 45, -1, -1, -1, -1, 45, 25],
                                      [-10, -1, -3, -2, -2, -3, -1, -10],
                                      [-5, -1, -2, -1, -1, -2, -1, -5],
                                      [-5, -1, -2, -1, -1, -2, -1, -5],
                                      [-10, -1, -3, -2, -2, -3, -1, -10],
                                      [25, 45, -1, -1, -1, -1, 45, 25],
                                      [-500, 25, -10, -5, -5, -10, 25, -500]])

    def go(self, chessboard):
        if self.beginning:
            self.beginning = self.judge_beginning(chessboard)
        self.candidate_list.clear()
        self.next_state.clear()
        self.next_state = self.get_state(chessboard, self.color)
        self.candidate_list = list(self.next_state.keys())
        if len(self.candidate_list) == 0:
            return self.candidate_list
        self.candidate_list.append(random.choice(self.candidate_list))
        count = self.count_chess(chessboard)
        if count < 12:
            pos = self.alpha_beta(chessboard, self.color)
        elif self.beginning and count < 18:
            pos = self.alpha_beta(chessboard, self.color, d=5)
        elif count < 52:
            pos = self.alpha_beta(chessboard, self.color)
        else:
            pos = self.alpha_beta(chessboard, self.color, d=8)
        if pos is None:
            return self.candidate_list
        self.candidate_list.pop()
        self.candidate_list.append(pos)
        return self.candidate_list

    def alpha_beta(self, chessboard, color, d=4):
        cache = self.load_cache(chessboard, color)
        if cache:
            return cache

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
        if best_action is not None:
            self.save_cache(chessboard=chessboard, color=color, pos=best_action)
        return best_action

    def evaluate(self, chessboard):
        def weighting(weighted_map):
            return np.sum(chessboard * weighted_map * self.color)

        cnt = self.count_chess(chessboard)
        if self.beginning and cnt < 20:
            return weighting(weighted_map=self.weighted_map0)
        elif cnt < 40:
            return weighting(weighted_map=self.weighted_map1)
        elif cnt < 55:
            return weighting(weighted_map=self.weighted_map2)
        else:
            return self.calculate_score(chessboard, self.color)

    def save_cache(self, chessboard, color, pos):
        key = self.get_key(chessboard, color)
        self.cache[key] = pos

    def load_cache(self, chessboard, color):
        key = self.get_key(chessboard, color)
        return self.cache.get(key)

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

    def judge_beginning(self, chessboard):
        for pos in self.x_squares + self.side_squares:
            if chessboard[pos[0]][pos[1]] != 0:
                return False
        return True

    @staticmethod
    def get_key(chessboard, color):
        board_map = chessboard.copy()
        board_map[board_map == color] = 1
        board_map[board_map == -color] = -1
        board = tuple([tuple(e) for e in board_map])
        key = tuple([board, color])
        return key

    @staticmethod
    @jit(nopython=True)
    def calculate_score(chessboard, color):
        black_num = np.sum(chessboard == COLOR_BLACK)
        white_num = np.sum(chessboard == COLOR_WHITE)
        return (white_num - black_num) * (-color) * 15000

    @staticmethod
    @jit(nopython=True)
    def count_chess(chessboard):
        return np.sum(chessboard != 0)
