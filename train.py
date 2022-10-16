class Gene:
    def __init__(self, data):
        self.ai = None
        self.size = len(data)


low = -500
high = 100
cr_p = 0.2
mut_p = 0.2

ngen = 100
pop_size = 24


for _ in range(pop_size):
    gene = []



