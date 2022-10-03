import copy
import math
import random
from functools import cmp_to_key
from concurrent.futures import *

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
        return -1


def cmp(t1: tuple, t2: tuple):
    num1 = tuple_to_int(t1)
    num2 = tuple_to_int(t2)
    return num1 - num2


class AI(object):

    def __init__(self, chessboard_size, color, time_out):
        self.chessboard_size = chessboard_size
        self.color = color
        self.time_out = time_out

        self.chessboard = None
        self.candidate_list = []
        self.next_state = {}

        self.precedence = (precedence0, precedence1, precedence2, precedence3, precedence4)
        self.cache = {}
        self.pool = ThreadPoolExecutor(max_workers=self.chessboard_size ** 2)

    def go(self, chessboard):
        self.candidate_list.clear()
        self.next_state.clear()
        self.chessboard = chessboard
        self.next_state = self.get_state(self.chessboard, self.color)
        self.candidate_list = list(self.next_state.keys())
        if len(self.candidate_list) == 0:
            return self.candidate_list
        except_corner = [candidate for candidate in self.candidate_list if candidate not in self.precedence[4]]
        if len(except_corner) == 0:
            pass
        else:
            pass
        return self.candidate_list

    def alpha_beta(self, chessboard, color, alpha, beta, depth, no_move=False):
        cache = self.load_cache(chessboard, color, alpha, beta)
        if cache:
            return cache
        saved_alpha, saved_beta = alpha, beta
        best_value = -math.inf
        best_pos = random.choice(self.candidate_list)
        state = self.get_state(chessboard, color)
        sorted_state = sorted(state, key=cmp_to_key(cmp))
        for pos in sorted_state:
            sub_board = state[pos]
            if depth == 0:
                value = -self.evaluate(sub_board, -color)
            else:
                value = -self.alpha_beta(sub_board, -color, -beta, -alpha, depth - 1)[0]
            if value >= beta:
                return value, pos
            if value > best_value:
                best_pos = pos
                best_value = value
                if value > alpha:
                    alpha = value
        if len(state) == 0:
            if no_move:
                self.calculate_score(chessboard, color)
            else:
                best_value = -self.alpha_beta(chessboard, -color, -beta, -alpha, depth, True)[0]
        self.save_cache(chessboard, color, saved_alpha, saved_beta, best_value, best_pos)
        return best_value, best_pos

    def evaluate(self, chessboard, color):
        return 0

    def save_cache(self, chessboard, color, alpha, beta, value, pos):
        key = self.get_key(alpha, beta, chessboard, color)
        self.cache[key] = (value, pos)

    def load_cache(self, chessboard, color, alpha, beta):
        key = self.get_key(alpha, beta, chessboard, color)
        return self.cache.get(key)

    def judge(self, x, y):
        return 0 <= x < self.chessboard_size and 0 <= y < self.chessboard_size

    def get_state(self, chessboard, color):

        def get_valid_pos(x, y, dx, dy):
            pos_set = []
            including = False
            while True:
                x += dx
                y += dy
                if not self.judge(x, y) or chessboard[x][y] == COLOR_NONE:
                    break
                elif chessboard[x][y] == color and not including:
                    break
                elif chessboard[x][y] == -color:
                    pos_set.append((x, y))
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
        with self.pool as t:
            obj_list = []
            for idx in indexes:
                obj = t.submit(test_all_directions, idx[0], idx[1])
                obj_list.append(obj)
            for obj in as_completed(obj_list):
                idx, reversed_color_set = obj.result()
                if len(reversed_color_set) > 0:
                    new_chessboard = copy.deepcopy(chessboard)
                    for pos in reversed_color_set:
                        new_chessboard[pos[0]][pos[1]] = color
                    new_chessboard[idx[0]][idx[1]] = color
                    next_state[idx] = new_chessboard
        return next_state

    @staticmethod
    def get_key(alpha, beta, chessboard, color):
        board_map = copy.deepcopy(chessboard)
        board_map[board_map == color] = 1
        board_map[board_map == -color] = -1
        board = tuple([tuple(e) for e in board_map])
        key = tuple([board, alpha, beta])
        return key

    @staticmethod
    def calculate_score(chessboard, color):
        black_num = np.where(chessboard == COLOR_BLACK)
        black_num = len(black_num[0])
        white_num = np.where(chessboard == COLOR_WHITE)
        white_num = len(white_num[0])
        return (white_num - black_num) * color * 20000
