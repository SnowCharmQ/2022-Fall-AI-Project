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

weighted_graph = [[90, -60, 10, 10, 10, 10, -60, 90],
                  [-60, -80, 5, 5, 5, 5, -80, -60],
                  [10, 5, 1, 1, 1, 1, 5, 10],
                  [10, 5, 1, 1, 1, 1, 5, 10],
                  [10, 5, 1, 1, 1, 1, 5, 10],
                  [10, 5, 1, 1, 1, 1, 5, 10],
                  [-60, -80, 5, 5, 5, 5, -80, -60],
                  [90, -60, 10, 10, 10, 10, -60, 90]]


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
        self.candidate_list = []
        self.next_state = {}
        self.precedence = (precedence0, precedence1, precedence2, precedence3, precedence4)
        self.cache = {}
        self.count = 0

    def go(self, chessboard):
        self.count += 1
        self.candidate_list.clear()
        self.next_state.clear()
        self.next_state = self.get_state(chessboard, self.color)
        self.candidate_list = list(self.next_state.keys())
        length = len(self.candidate_list)
        if length == 0:
            return self.candidate_list
        for choice in self.candidate_list:
            if choice not in self.precedence[4]:
                self.candidate_list.append(choice)
                break
        if length != 0 and len(self.candidate_list) == length:
            self.candidate_list.append(random.choice(self.candidate_list))
        if self.color == COLOR_BLACK:
            depth = 3
        else:
            depth = 3
        val, pos = self.alpha_beta(chessboard, self.color, -10000, -10000, depth)
        self.candidate_list.pop()
        self.candidate_list.append(pos)
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
            return current_color, frontier, stability

        weight = 0
        my_frontier = 0
        op_frontier = 0
        my_stability = 0
        op_stability = 0
        state = self.get_state(chessboard, color)
        my_corner_choice = [c for c in self.precedence[4] if chessboard[c[0]][c[1]] != color]
        my_corner_choice = len(my_corner_choice)
        my_mobility = len(state) - my_corner_choice
        for i in range(self.chessboard_size):
            for j in range(self.chessboard_size):
                item = chessboard[i][j]
                weight += item * weighted_graph[i][j]
                result = calculate(chessboard, item, i, j)
                if result[0] == color:
                    my_frontier += result[1]
                    my_stability += result[2]
                else:
                    op_frontier += result[1]
                    op_stability += result[2]
        if color == COLOR_WHITE:
            weight = -weight
        if len(state) != 0:
            return weight - 16 * (my_stability - op_stability) + 6 * (
                        my_frontier - op_frontier) - 8 * my_mobility - 100 * my_corner_choice / len(state)
        else:
            return weight * 2 - 16 * (my_stability - op_stability) + 6 * (
                    my_frontier - op_frontier) - 8 * my_mobility

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
        with ThreadPoolExecutor(max_workers=self.chessboard_size ** 2) as t:
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
        return (white_num - black_num) * color * 10000
