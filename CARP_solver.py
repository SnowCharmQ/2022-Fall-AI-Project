import copy
import time
import random
import argparse
import threading
import numpy as np

start_time = time.time()
params = dict()
num_node, num_edge = 0, 0
edges, graph, distances, demands = set(), [[0]], [[0]], [[0]]

parser = argparse.ArgumentParser()
parser.add_argument("file_path", help="the absolute path of the test CARP instance file")
parser.add_argument("-t", help="the termination", type=int)
parser.add_argument("-s", help="random seed", type=int)

args = parser.parse_args()

file_path = args.file_path
t = args.t
s = args.s
random.seed(s)

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
        graph = [[np.inf] * (num_node + 1) for _ in range(num_node + 1)]
        distances = [[np.inf] * (num_node + 1) for _ in range(num_node + 1)]
        demands = [[0] * (num_node + 1) for _ in range(num_node + 1)]
    elif 9 <= i <= 8 + num_edge:
        nums = line.split()
        num1, num2, cost, demand = int(nums[0]), int(nums[1]), int(nums[2]), int(nums[3])
        edges.add((num1, num2))
        edges.add((num2, num1))
        graph[num1][num2], graph[num2][num1] = cost, cost
        distances[num1][num2], distances[num2][num1] = cost, cost
        demands[num1][num2], demands[num2][num1] = demand, demand
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


def rule1(dis, arc, depot, cap):
    return distances[depot][arc[0]] > dis


def rule2(dis, arc, depot, cap):
    return distances[depot][arc[0]] < dis


def rule3(dis, arc, depot, cap):
    return demands[arc[0]][arc[1]] / graph[arc[0]][arc[1]] > dis


def rule4(dis, arc, depot, cap):
    return demands[arc[0]][arc[1]] / graph[arc[0]][arc[1]] < dis


class RuleThread(threading.Thread):
    def __init__(self, free, depot, capacity, rule: str):
        super().__init__()
        self.free = free
        self.depot = depot
        self.capacity = capacity
        self.routes = []
        self.total_cost = 0
        self.total_load = 0
        if rule == 'rule1':
            self.rule = rule1
        elif rule == 'rule2':
            self.rule = rule2
        elif rule == 'rule3':
            self.rule = rule3
        elif rule == 'rule4':
            self.rule = rule4

        k = 0
        while len(self.free) > 0:
            route = []
            k += 1
            load, cost = 0, 0
            i = self.depot
            while True:
                arc = None
                dis = np.inf
                for u in self.free:
                    if load + demands[u[0]][u[1]] <= capacity:
                        if self.rule(dis, u, i, cost):
                            dis = distances[i][u[0]]
                            arc = u
                if dis == np.inf:
                    break
                route.append(arc)
                free.remove(arc)
                free.remove((arc[1], arc[0]))
                load += demands[arc[0]][arc[1]]
                cost += (distances[arc[0]][arc[1]] + graph[arc[0]][arc[1]])
                i = arc[1]
            cost += distances[i][depot]
            self.total_cost += cost
            self.total_load += load
            self.routes.append(route)


k = 0
depot = int(params['DEPOT'])
capacity = int(params['CAPACITY'])
free = copy.deepcopy(edges)


def print_info(routes, total_cost):
    print("s ", end="")
    for i in range(len(routes)):
        route = routes[i]
        print(0, end="")
        print(",", end="")
        for arc in route:
            print(str(arc).replace(' ', ''), end="")
            print(",", end="")
        print(0, end="")
        if i != len(routes) - 1:
            print(",", end="")
        else:
            print()
    print("q %d" % total_cost)
