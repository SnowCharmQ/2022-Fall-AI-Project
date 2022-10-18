from ai import *
from idiot import *
from concurrent.futures import *


class Gene:
    def __init__(self, data):
        self.data = data
        self.size = len(data)
        self.fitness = 0

    def __lt__(self, other):
        return self.fitness < other.fitness


def evaluate(g1: Gene, g2: Gene):
    cnt = 0
    ai_black = AI(8, COLOR_BLACK, 5, g1.data)
    ai_white = AI(8, COLOR_WHITE, 5, g2.data)
    cnt += battle_against(ai_black, ai_white)
    ai_black = AI(8, COLOR_BLACK, 5, g2.data)
    ai_white = AI(8, COLOR_WHITE, 5, g1.data)
    cnt -= battle_against(ai_black, ai_white)
    if cnt == 2:
        g1.fitness += 10
    elif cnt == 1:
        g1.fitness += 7
        g2.fitness += 2
    elif cnt == -1:
        g2.fitness += 7
        g1.fitness += 2
    elif cnt == -2:
        g2.fitness += 10
    else:
        g1.fitness += 4
        g2.fitness += 4


def get_valid_gene():
    while 1:
        data = []
        for pos in range(len(low) - 2):
            data.append(random.randint(bound[0][pos], bound[1][pos]))
        data.append(random.random())
        data.append(random.random())
        g = Gene(data)
        cnt = 0
        ai_black = AI(8, COLOR_BLACK, 5, g.data)
        ai_white = Idiot(8, COLOR_WHITE, 5)
        cnt += battle_against(ai_black, ai_white)
        ai_black = Idiot(8, COLOR_BLACK, 5)
        ai_white = AI(8, COLOR_WHITE, 5, g.data)
        cnt -= battle_against(ai_black, ai_white)
        if cnt == 2:
            return g


def get_winner(chessboard):
    black = np.sum(chessboard == COLOR_BLACK)
    white = np.sum(chessboard == COLOR_WHITE)
    if black > white:
        return 1
    elif black < white:
        return -1
    else:
        return 0


def judge(chessboard):
    if np.sum(chessboard != 0) == 64:
        return True
    return False


def battle_against(ai_black, ai_white):
    board = np.zeros((8, 8))
    board[3][3] = COLOR_WHITE
    board[4][4] = COLOR_WHITE
    board[3][4] = COLOR_BLACK
    board[4][3] = COLOR_BLACK
    flag = False
    while True:
        result = ai_black.go(board)
        if len(result) > 0:
            piece = result[-1]
            board = ai_black.next_state[piece]
            if flag:
                flag = False
        else:
            if flag:
                break
            flag = True
        if judge(board):
            break
        result = ai_white.go(board)
        if len(result) > 0:
            piece = result[-1]
            board = ai_white.next_state[piece]
            if flag:
                flag = False
        else:
            if flag:
                break
            flag = True
        if judge(board):
            break
    winner = get_winner(chessboard=board)
    if winner == COLOR_BLACK:
        return 1
    elif winner == COLOR_WHITE:
        return -1
    else:
        return 0


def get_fitness(population: list):
    with ThreadPoolExecutor(max_workers=100) as t:
        obj_list = []
        pair_set = set()
        for g1 in population:
            for g2 in population:
                if (g1, g2) not in pair_set and (g2, g1) not in pair_set and g1 != g2:
                    pair_set.add((g1, g2))
                    obj = t.submit(evaluate, g1, g2)
                    obj_list.append(obj)
        for obj in as_completed(obj_list):
            obj.result()


low = -500
low = [low for _ in range(30)]
low.extend([0, 0, low, 1])
high = 100
high = [high for _ in range(30)]
high.extend([1, 1, high, 10])
bound = (low, high)
cr_p = 0.2
mut_p = 0.2
ngen = 100

pop = []
pop_size = 16

init_data = [-500, 25, -10, 45, -1, -5, -3, -2, -1, -1,
             -500, 25, -10, 45, -1, -5, -3, -2, -1, -1,
             -500, 25, -10, 45, -1, -5, -3, -2, -1, -1,
             1, 1, 0.1, 0.5]
init_gene = Gene(init_data)
pop.append(init_gene)
init_data = [-70, 5, -2, 10, -5, -2, -1, -1, -1, -2,
             -70, 5, -2, 10, -5, -2, -1, -1, -1, -2,
             -70, 5, -2, 10, -5, -2, -1, -1, -1, -2,
             1, 1, 0.05, 0.5]
init_gene = Gene(init_data)
pop.append(init_gene)
s = time.time()
with ThreadPoolExecutor(max_workers=100) as t:
    obj_list = []
    for _ in range(pop_size - 2):
        obj = t.submit(get_valid_gene)
        obj_list.append(obj)
    for obj in as_completed(obj_list):
        result = obj.result()
        pop.append(result)
get_fitness(pop)
e = time.time()
print(e - s)
pop.sort(reverse=True)
