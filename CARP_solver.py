import copy
import math
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
termination = args.t
seed = args.s
random.seed(seed)

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


def judge(flag=False):
    if not flag:
        weight = [0.3, 0.6, 1]
        random_num = random.random()
        if random_num < weight[0]:
            operation = flip
        elif random_num < weight[1]:
            operation = self_swap
        else:
            operation = cross_swap
        return operation
    else:
        return combination


def rule1(dis, arc, depot, cap):
    return distances[depot][arc[0]] > dis


def rule2(dis, arc, depot, cap):
    return distances[depot][arc[0]] < dis


def rule3(dis, arc, depot, cap):
    return demands[arc[0]][arc[1]] / graph[arc[0]][arc[1]] > dis


def rule4(dis, arc, depot, cap):
    return demands[arc[0]][arc[1]] / graph[arc[0]][arc[1]] < dis


def rule5(dis, arc, depot, cap):
    if cap < capacity / 2:
        return rule1(dis, arc, depot, cap)
    else:
        return rule2(dis, arc, depot, cap)


def probability(new_val, old_val, T):
    if new_val < old_val:
        return 1
    return math.exp((old_val - new_val) / T)


def flip(routes, costs, route_demands, depot, best_routes, best_costs, best_demands):
    for l in range(len(routes)):
        route = routes[l]
        for i in range(len(route)):
            if random.random() < 0.5:
                if i == 0:
                    start = depot
                else:
                    start = route[i - 1][1]
                if i == len(route) - 1:
                    end = depot
                else:
                    end = route[i + 1][0]
                costs[l] = costs[l] - distances[start][route[i][0]] - distances[route[i][1]][end] + \
                           distances[start][route[i][1]] + distances[route[i][0]][end]
                route[i] = (route[i][1], route[i][0])
                routes[l] = route
                current_cost = sum(costs)
                if current_cost < sum(best_costs):
                    best_costs = copy.deepcopy(costs)
                    best_routes = copy.deepcopy(routes)
                    best_demands = copy.deepcopy(route_demands)
    return routes, costs, route_demands, best_routes, best_costs, best_demands


def self_swap(routes, costs, route_demands, depot, best_routes, best_costs, best_demands):
    for l in range(len(routes)):
        route = routes[l]
        if len(route) == 1 or random.random() < 0.5:
            continue
        pos1 = 0
        pos2 = 0
        while pos1 == pos2:
            pos1 = random.randint(0, len(route) - 1)
            pos2 = random.randint(0, len(route) - 1)
        if pos1 > pos2:
            pos1, pos2 = pos2, pos1
        seg1 = route[pos1]
        seg2 = route[pos2]
        if pos1 == 0:
            start1 = depot
        else:
            start1 = route[pos1 - 1][1]
        if pos2 == 0:
            start2 = depot
        else:
            start2 = route[pos2 - 1][1]
        if pos1 == len(route) - 1:
            end1 = depot
        else:
            end1 = route[pos1 + 1][0]
        if pos2 == len(route) - 1:
            end2 = depot
        else:
            end2 = route[pos2 + 1][0]
        route[pos1], route[pos2] = seg2, seg1
        routes[l] = route
        if pos2 - pos1 != 1:
            temp = costs[l] - distances[start1][seg1[0]] - distances[seg1[1]][end1] - \
                   distances[start2][seg2[0]] - distances[seg2[1]][end2]
            temp += (distances[start1][seg2[0]] + distances[seg2[1]][end1] +
                     distances[start2][seg1[0]] + distances[seg1[1]][end2])
        else:
            temp = costs[l] - distances[start1][seg1[0]] - distances[seg2[1]][end2] - \
                   distances[seg1[1]][seg2[0]]
            temp += (distances[start1][seg2[0]] + distances[seg1[1]][end2] +
                     distances[seg2[1]][seg1[0]])
        costs[l] = temp
        current_cost = sum(costs)
        if current_cost < sum(best_costs):
            best_costs = copy.deepcopy(costs)
            best_routes = copy.deepcopy(routes)
            best_demands = copy.deepcopy(route_demands)
    return routes, costs, route_demands, best_routes, best_costs, best_demands


def cross_swap(routes, costs, route_demands, depot, best_routes, best_costs, best_demands, swap=5):
    if len(routes) == 1:
        return routes, costs, route_demands, best_routes, best_costs, best_demands
    num = 0
    while num < swap:
        num += 1
        r1 = 0
        r2 = 0
        while r1 == r2:
            r1 = random.randint(0, len(routes) - 1)
            r2 = random.randint(0, len(routes) - 1)
        if len(routes[r1]) == 1 and len(routes[r2]) == 1:
            continue
        cnt = 0
        while True:
            cnt += 1
            demand1, demand2 = route_demands[r1], route_demands[r2]
            pos1 = random.randint(0, len(routes[r1]) - 1)
            pos2 = random.randint(0, len(routes[r2]) - 1)
            seg1 = routes[r1][pos1]
            seg2 = routes[r2][pos2]
            new_demand1 = demand1 - demands[seg1[0]][seg1[1]] + demands[seg2[0]][seg2[1]]
            new_demand2 = demand2 - demands[seg2[0]][seg2[1]] + demands[seg1[0]][seg1[1]]
            if cnt >= len(routes[r1]) + len(routes[r2]):
                break
            if new_demand1 <= capacity and new_demand2 <= capacity:
                routes[r1][pos1], routes[r2][pos2] = seg2, seg1
                route_demands[r1], route_demands[r2] = new_demand1, new_demand2
                if pos1 == 0:
                    start1 = depot
                else:
                    start1 = routes[r1][pos1 - 1][1]
                if pos1 == len(routes[r1]) - 1:
                    end1 = depot
                else:
                    end1 = routes[r1][pos1 + 1][0]
                if pos2 == 0:
                    start2 = depot
                else:
                    start2 = routes[r2][pos2 - 1][1]
                if pos2 == len(routes[r2]) - 1:
                    end2 = depot
                else:
                    end2 = routes[r2][pos2 + 1][0]
                costs[r1] = costs[r1] - distances[start1][seg1[0]] - distances[seg1[1]][end1] - \
                            graph[seg1[0]][seg1[1]] + distances[start1][seg2[0]] + distances[seg2[1]][end1] + \
                            graph[seg2[0]][seg2[1]]
                costs[r2] = costs[r2] - distances[start2][seg2[0]] - distances[seg2[1]][end2] - \
                            graph[seg2[0]][seg2[1]] + distances[start2][seg1[0]] + distances[seg1[1]][end2] + \
                            graph[seg1[0]][seg1[1]]
                current_cost = sum(costs)
                if current_cost < sum(best_costs):
                    best_costs = copy.deepcopy(costs)
                    best_routes = copy.deepcopy(routes)
                    best_demands = copy.deepcopy(route_demands)
    return routes, costs, route_demands, best_routes, best_costs, best_demands


def combination(routes, costs, route_demands, depot, best_routes, best_costs, best_demands):
    routes, costs, route_demands, best_routes, best_costs, best_demands = flip(routes, costs, route_demands, depot,
                                                                               best_routes, best_costs, best_demands)
    routes, costs, route_demands, best_routes, best_costs, best_demands = flip(routes, costs, route_demands, depot,
                                                                               best_routes, best_costs, best_demands)
    routes, costs, route_demands, best_routes, best_costs, best_demands = self_swap(routes, costs, route_demands, depot,
                                                                                    best_routes, best_costs,
                                                                                    best_demands)
    routes, costs, route_demands, best_routes, best_costs, best_demands = self_swap(routes, costs, route_demands, depot,
                                                                                    best_routes, best_costs,
                                                                                    best_demands)
    routes, costs, route_demands, best_routes, best_costs, best_demands = cross_swap(routes, costs,
                                                                                     route_demands, depot,
                                                                                     best_routes, best_costs,
                                                                                     best_demands, 25)
    return routes, costs, route_demands, best_routes, best_costs, best_demands


class RuleThread(threading.Thread):
    def __init__(self, free, depot, capacity, rule: str):
        super().__init__()
        self.free = free
        self.depot = depot
        self.capacity = capacity
        self.routes = []
        self.costs = []
        self.route_demands = []
        self.total_cost = 0
        self.total_load = 0
        self.protocol = rule
        if rule == 'rule1':
            self.rule = rule1
        elif rule == 'rule2':
            self.rule = rule2
        elif rule == 'rule3':
            self.rule = rule3
        elif rule == 'rule4':
            self.rule = rule4
        elif rule == 'rule5':
            self.rule = rule5
        else:
            self.rule = None

    def run(self):
        while self.free:
            route = []
            load, cost = 0, 0
            i = self.depot
            while True:
                arc = None
                dis = np.inf
                if self.protocol == 'rule5' and load < capacity / 2:
                    dis = -1
                elif self.protocol == 'rule5':
                    dis = np.inf
                elif self.protocol == 'rule1' or self.protocol == 'rule3':
                    dis = -1
                elif self.protocol == 'rule2' or self.protocol == 'rule4':
                    dis = np.inf
                choice = []
                for u in self.free:
                    if load + demands[u[0]][u[1]] <= capacity:
                        if self.rule and self.rule(dis, u, i, cost):
                            if self.protocol == 'rule1' or self.protocol == 'rule2' or self.protocol == 'rule5':
                                dis = distances[i][u[0]]
                            else:
                                dis = demands[u[0]][u[1]] / graph[u[0]][u[1]]
                            arc = u
                        elif not self.rule:
                            if distances[i][u[0]] < dis:
                                dis = distances[i][u[0]]
                                choice.clear()
                                choice.append(u)
                            elif distances[i][u[0]] == dis:
                                choice.append(u)
                if len(choice) > 0:
                    arc = random.choice(choice)
                if dis == -1 or dis == np.inf or arc is None:
                    break
                route.append(arc)
                self.free.remove(arc)
                self.free.remove((arc[1], arc[0]))
                load += demands[arc[0]][arc[1]]
                cost += (distances[i][arc[0]] + graph[arc[0]][arc[1]])
                i = arc[1]
            cost += distances[i][depot]
            self.total_cost += cost
            self.total_load += load
            self.route_demands.append(load)
            self.costs.append(cost)
            self.routes.append(route)
        T = 10000
        alpha = 0.99
        routes = copy.deepcopy(self.routes)
        costs = copy.deepcopy(self.costs)
        route_demands = copy.deepcopy(self.route_demands)
        repeat = 0
        last_routes = copy.deepcopy(routes)
        last_cost = sum(costs)
        flag = False
        while time.time() - start_time <= termination - 2:
            operation = judge(flag)
            if flag:
                flag = not flag
            temp_routes, temp_costs, temp_route_demands, \
            best_routes, best_costs, best_route_demands = operation(routes, costs,
                                                                    route_demands,
                                                                    self.depot,
                                                                    self.routes,
                                                                    self.costs,
                                                                    self.route_demands)
            self.routes = copy.deepcopy(best_routes)
            self.costs = copy.deepcopy(best_costs)
            self.route_demands = copy.deepcopy(best_route_demands)
            temp_routes, temp_costs, temp_route_demands = \
                copy.deepcopy(temp_routes), copy.deepcopy(temp_costs), copy.deepcopy(temp_route_demands)
            if probability(sum(temp_costs), sum(costs), T) > random.random():
                routes = copy.deepcopy(temp_routes)
                costs = copy.deepcopy(temp_costs)
                route_demands = copy.deepcopy(temp_route_demands)
            if last_cost == sum(costs) or routes == last_routes:
                repeat += 1
            else:
                last_cost = sum(costs)
                last_routes = copy.deepcopy(routes)
            if sum(costs) > 1.2 * sum(self.costs):
                T = 10000
                routes = copy.deepcopy(self.routes)
                costs = copy.deepcopy(self.costs)
                route_demands = copy.deepcopy(self.route_demands)
                last_routes = copy.deepcopy(self.routes)
                flag = True
                continue
            if repeat > 100:
                last_routes = copy.deepcopy(routes)
                T = 10000
                repeat = 0
                flag = True
                continue
            T *= alpha
        self.total_cost = sum(self.costs)


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


k = 0
depot = int(params['DEPOT'])
capacity = int(params['CAPACITY'])

thread_list = []
t1 = RuleThread(copy.deepcopy(edges), depot, capacity, "rule1")
t2 = RuleThread(copy.deepcopy(edges), depot, capacity, "rule2")
t3 = RuleThread(copy.deepcopy(edges), depot, capacity, "rule3")
t4 = RuleThread(copy.deepcopy(edges), depot, capacity, "rule4")
t5 = RuleThread(copy.deepcopy(edges), depot, capacity, "rule5")
t6 = RuleThread(copy.deepcopy(edges), depot, capacity, "random")
t7 = RuleThread(copy.deepcopy(edges), depot, capacity, "random")
thread_list.append(t1)
thread_list.append(t2)
thread_list.append(t3)
thread_list.append(t4)
thread_list.append(t5)
thread_list.append(t6)
thread_list.append(t7)

final_cost = np.inf
final_routes = []
for t in thread_list:
    t.start()
for t in thread_list:
    t.join()
for t in thread_list:
    if t.total_cost < final_cost:
        final_cost = t.total_cost
        final_routes = t.routes

print_info(final_routes, final_cost)
