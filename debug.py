from main import *

board_s = """
[[ 0  1  1  1  1  0  0  0]
 [-1 -1  1  1  1 -1  0  0]
 [-1  1  1  1  1  0  0  0]
 [-1  0 -1 -1  1 -1  0  0]
 [ 0  0  0 -1  1  0  0  0]
 [ 0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0]]
 """

board_s = board_s.replace("[", "").replace("]", "").replace("\n", "")
board_l = board_s.split(" ")
board_l = list(filter(lambda x: x != "", board_l))
board_l = list(map(lambda x: int(x), board_l))
board_np = np.array(board_l).reshape((8, 8))
ai = AI(8, -1, 5)
ai.go(board_np)
print(ai.candidate_list)
