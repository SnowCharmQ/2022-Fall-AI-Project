import math
import random
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
        self.count = 0
        self.beginning = True

        self.inner_squares = [(3, 3), (4, 4), (4, 3), (3, 4)]
        self.x_squares = [(1, 1), (1, 6), (6, 1), (6, 6)]
        self.c_squares = [(0, 1), (0, 6), (1, 0), (1, 7), (6, 0), (6, 7), (7, 1), (7, 6)]
        self.corner_squares = [(0, 0), (0, 7), (7, 0), (7, 7)]
        self.side_squares = [(x, y) for x in range(8) for y in range(8) if (x == 0 or x == 7 or y == 0 or y == 7)]

        self.where = np.where

    def go(self, chessboard):
        if self.beginning:
            self.beginning = self.judge_beginning(chessboard)
        self.candidate_list.clear()
        self.next_state.clear()
        self.next_state = self.get_state(chessboard, self.color)
        self.candidate_list = list(self.next_state.keys())
        self.count = self.count_chess(chessboard)
        if len(self.candidate_list) == 0:
            return self.candidate_list
        self.candidate_list.append(random.choice(self.candidate_list))
        if self.color == COLOR_BLACK:
            if self.beginning and self.count <= 6:
                val, pos = self.alpha_beta(chessboard, self.color, 3)
            elif self.count <= 12:
                val, pos = self.alpha_beta(chessboard, self.color, 4)
            elif 12 < self.count < 30:
                val, pos = self.alpha_beta(chessboard, self.color, 3)
            else:
                val, pos = self.alpha_beta(chessboard, self.color, 6)
        else:
            if self.beginning and self.count <= 6:
                val, pos = self.alpha_beta(chessboard, self.color, 3)
            elif self.count <= 12:
                val, pos = self.alpha_beta(chessboard, self.color, 3)
            elif 12 < self.count < 30:
                val, pos = self.alpha_beta(chessboard, self.color, 3)
            else:
                val, pos = self.alpha_beta(chessboard, self.color, 6)
        print(self.color, val)
        if pos is None:
            return self.candidate_list
        self.candidate_list.pop()
        self.candidate_list.append(pos)
        return self.candidate_list

    def alpha_beta(self, chessboard, color, depth):
        cache = self.load_cache(chessboard, color)
        if cache:
            return cache

        def maxvalue(board, alpha, beta, current_color, depth_val, no_move=False):
            state = self.get_state(board, current_color)
            if depth_val == 0 and len(state) == 0:
                return self.calculate_score(board, self.color), None
            elif depth_val == 0:
                return self.evaluate(board, self.color), None
            elif len(state) == 0:
                if no_move:
                    return self.calculate_score(board, self.color), None
                else:
                    return minvalue(board, alpha, beta, -current_color, depth_val - 1, True)
            best_value, best_move = -math.inf, random.choice(self.candidate_list)
            sorted_state = sorted(state, key=cmp_to_key(cmp))
            get = state.get
            for pos in sorted_state:
                sub_board = get(pos)
                value, move = minvalue(sub_board, alpha, beta, -current_color, depth_val - 1)
                if value > best_value:
                    best_value = value
                    best_move = pos
                    alpha = max(alpha, best_value)
                if alpha >= beta:
                    return best_value, best_move
            return best_value, best_move

        def minvalue(board, alpha, beta, current_color, depth_val, no_move=False):
            state = self.get_state(board, current_color)
            if depth_val == 0 and len(state) == 0:
                return self.calculate_score(board, self.color), None
            elif depth_val == 0:
                return self.evaluate(board, self.color), None
            elif len(state) == 0:
                if no_move:
                    return self.calculate_score(board, self.color), None
                else:
                    return maxvalue(board, alpha, beta, -current_color, depth_val - 1, True)
            best_value, best_move = math.inf, None
            sorted_state = sorted(state, key=cmp_to_key(cmp))
            get = state.get
            for pos in sorted_state:
                sub_board = get(pos)
                value, move = maxvalue(sub_board, alpha, beta, -current_color, depth_val - 1)
                if value < best_value:
                    best_value = value
                    best_move = pos
                    beta = min(beta, best_value)
                if alpha >= beta:
                    return best_value, best_move
            return best_value, best_move

        return maxvalue(board=chessboard, alpha=-math.inf, beta=math.inf, current_color=color, depth_val=depth)

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
            if is_frontier(board, x, y):
                frontier += 1
            stable_list = list(filter(
                lambda num: True if judge_stable(board, current_color, x, y, num) and judge_stable(board, current_color,
                                                                                                   x, y,
                                                                                                   num + 1) else False,
                [0, 2, 4, 6]))
            return frontier, len(stable_list)

        my_frontier = 0
        my_stability = 0
        op_frontier = 0
        op_stability = 0
        state_score = 0
        for i in range(8):
            for j in range(8):
                if chessboard[i][j] == color:
                    result = calculate(chessboard, color, i, j)
                    my_frontier += result[0]
                    my_stability += result[1]
                    state_score += self.map_to_state((i, j))
                elif chessboard[i][j] == -color:
                    result = calculate(chessboard, -color, i, j)
                    op_frontier += result[0]
                    op_stability += result[1]
                    state_score -= self.map_to_state((i, j))
        next_state = self.get_state(chessboard, -color)
        mobi_score = 0
        if len(next_state) > 0:
            op_corner_cnt = 0
            for pos in next_state.keys():
                if pos in self.corner_squares:
                    op_corner_cnt += 1
            if op_corner_cnt == len(next_state):
                mobi_score += 12
            else:
                mobi_score += (2 * (len(next_state) - op_corner_cnt) + 5 * op_corner_cnt)
        if self.beginning:
            return state_score * 1.1 + mobi_score * 1.4 + (op_frontier - my_frontier) * 1.2 + (
                    op_stability - my_stability) * 1.5
        elif self.count <= 17:
            return state_score * 0.9 + mobi_score * 1.6 + (op_frontier - my_frontier) * 1.4 + (
                    op_stability - my_stability) * 1.6
        elif self.count <= 24:
            return state_score * 1.8 + mobi_score * 1.5 + (op_frontier - my_frontier) * 1.2 + (
                    op_stability - my_stability) * 1.3
        else:
            return len(tuple(zip(self.where(chessboard == -color)))) - len(tuple(zip(self.where(chessboard == color))))

    def save_cache(self, chessboard, color, value, pos):
        key = self.get_key(chessboard, color)
        self.cache[key] = (value, pos)

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

    def map_to_state(self, t: tuple):
        def func1(x):
            return -2 if x < 30 else -1

        def func2(x):
            return 1 if x < 30 else 0

        if t in self.inner_squares + self.precedence[3]:
            return func1(self.count)
        elif t in self.corner_squares:
            return -7
        elif t in self.x_squares:
            return 4
        elif t in self.c_squares:
            return -3
        elif t in self.precedence[2]:
            return func2(self.count)
        elif t in self.precedence[1]:
            return 2
        else:
            return 0

    def calculate_score(self, chessboard, color):
        black_num = self.where(chessboard == COLOR_BLACK)
        black_num = len(black_num[0])
        white_num = self.where(chessboard == COLOR_WHITE)
        white_num = len(white_num[0])
        return (white_num - black_num) * (-color) * 10000

    def judge_beginning(self, chessboard):
        for pos in self.x_squares + self.side_squares:
            if chessboard[pos[0]][pos[1]] != 0:
                return False
        return True

    def count_chess(self, chessboard):
        idx = self.where(chessboard != 0)
        return len(tuple(zip(idx[0], idx[1])))

    @staticmethod
    def get_key(chessboard, color):
        board_map = chessboard.copy()
        board_map[board_map == color] = 1
        board_map[board_map == -color] = -1
        board = tuple([tuple(e) for e in board_map])
        key = tuple([board])
        return key
