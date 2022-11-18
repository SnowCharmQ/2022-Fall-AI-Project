import time
import math
import random
import argparse
import numpy as np


class Edge:
    def __init__(self, n1, n2, c, d):
        self.num1 = n1
        self.num2 = n2
        self.cost = c
        self.demand = d
        self.max_dis = np.inf
        self.min_dis = -np.inf


def get_hash(n1: int, n2: int):
    if n1 > n2:
        n1, n2 = n2, n1
    return hash(str(n1) + " " + str(n2))


parser = argparse.ArgumentParser()
parser.add_argument("file_path", help="the absolute path of the test CARP instance file")
parser.add_argument("-t", help="the termination", type=int)
parser.add_argument("-s", help="random seed", type=int)

args = parser.parse_args()

file_path = args.file_path
t = args.t
s = args.s

params = dict()
num_node, num_edge = 0, 0
edges = {}
distances = [[]]

with open(file_path, 'r') as f:
    lines = f.readlines()

for i in range(len(lines)):
    line = lines[i].replace("\n", "")
    if 0 <= i <= 7:
        kv = line.split(" : ")
        params[kv[0]] = kv[1]
    elif i == 8:
        num_edge = int(params['REQUIRED EDGES']) + int(params['NON-REQUIRED EDGES'])
        num_node = int(params['VERTICES'])
        distances = [[np.inf] * (num_node + 1) for _ in range(num_node + 1)]
    elif 9 <= i <= 8 + num_edge:
        nums = line.split()
        num1, num2, cost, demand = int(nums[0]), int(nums[1]), int(nums[2]), int(nums[3])
        edges[get_hash(num1, num2)] = Edge(num1, num2, cost, demand)
        distances[num1][num2], distances[num2][num1] = cost, cost
    else:
        break

for n in range(1, num_node + 1):
    distances[n][n] = 0
for k in range(1, num_node + 1):
    for i in range(1, num_node + 1):
        for j in range(1, num_node + 1):
            temp = distances[i][k] + distances[k][j]
            if temp < distances[i][j]:
                distances[i][j] = temp

