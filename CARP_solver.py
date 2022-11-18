import time
import math
import random
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("file_path", help="the absolute path of the test CARP instance file")
parser.add_argument("-t", help="the termination", type=int)
parser.add_argument("-s", help="random seed", type=int)

args = parser.parse_args()

file_path = args.file_path
t = args.t
s = args.s

print(file_path)
print(t)
print(s)
