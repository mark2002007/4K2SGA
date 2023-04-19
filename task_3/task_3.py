import numpy as np
np.set_printoptions(linewidth=np.inf, suppress=True)
import matplotlib.pyplot as plt
#from operator import add, sub, mul, truediv as div
#from math import sqrt, sin, cos, pi
from math import pi
import copy

#Functions
def add(a, b) : return a + b
def sub(a, b) : return a - b
def mul(a, b) : return a * b
def div(a, b) : return 1e+8 if not b else a / b
def sqrt(a)   : return a**(1/2)
def sin(a)    : return np.sin(a)
def cos(a)    : return np.cos(a)
def abs(a)    : return np.abs(a) 

#Trees
class Node:
    def __init__(self, val=None, name=None, height=None, first_arg=None, second_arg=None):
        self.val = val
        self.name = name
        self.height = height
        self.first_arg = first_arg
        self.second_arg = second_arg

    def traversal(self): 
        ret = [self] 
        ret += ["(" , *self.first_arg.traversal() ] if self.first_arg  else []
        ret += [", ", *self.second_arg.traversal()] if self.second_arg else []
        ret += [")"]                                if self.first_arg  else []
        return ret
    
    def __str__(self):
        return "".join([n.name if isinstance(n, Node) else n for n in self.traversal()])
    
    def copy(self): 
        return copy.deepcopy(self)

    def load(self, node):
        self.val = node.val
        self.name = node.name
        self.height = node.height
        self.first_arg = node.first_arg
        self.second_arg = node.second_arg

    def compute(self, **subs): 
        args = []
        args += [self.first_arg.compute(**subs) ] if self.first_arg  else []
        args += [self.second_arg.compute(**subs)] if self.second_arg else []
        return self.val(*args) if args                    else \
               subs[self.val]  if self.val in subs.keys() else \
               self.val                                          

#Helpers
def rand_tree(conf, h = 1):
    min_depth, max_depth, term_prob, var_prob = conf.min_depth, conf.max_depth, conf.term_prob, conf.var_prob

    term_prob = 0 if h < min_depth else term_prob
    term_prob = 1 if h >= max_depth else term_prob
    if np.random.rand() >= term_prob:
        op = conf.ops[np.random.randint(len(conf.ops))]
        return Node(op, op.__name__, h, rand_tree(conf, h + 1)) if op.__code__.co_argcount == 1 else \
               Node(op, op.__name__, h, rand_tree(conf, h + 1), rand_tree(conf, h + 1))
    elif np.random.rand() < var_prob:
        var = conf.vars[np.random.randint(len(conf.vars))]
        return Node(var, var, h)
    else:
        const = conf.consts[np.random.randint(len(conf.consts))]
        return Node(const, str(const), h)

def select_node(I):
    return np.random.choice([node for node in I.traversal() if node not in ["(", ")", ", "]])

#GA
def select(P, P_eval):
    P = P[P_eval.argsort()]
    P_eval = P_eval[P_eval.argsort()]
    e = P_eval[-1] - P_eval + 1
    probs = e / np.sum(e)
    return np.random.choice(P, 2, False, probs) 

def crossover(I1, I2):
    I1_new, I2_new = I1.copy(), I2.copy()
    node_1, node_2 = select_node(I1_new), select_node(I2_new)
    node_t = Node()
    node_t.load(node_1)
    node_1.load(node_2)
    node_2.load(node_t)
    return I1_new, I2_new

def mutate(I, conf):
    new_I = I.copy()
    node = select_node(new_I)
    rt = rand_tree(conf, node.height)
    node.load(rt)
    return new_I

def f(I, conf):
    x_, y_ = conf.points
    y_eval = [I.compute(x = xx) for xx in x_]
    return np.sum((y_eval - y_)**2)

def ga(conf): 
    #
    P = np.array([rand_tree(conf) for i in range(conf.P_SIZE)])
    P_eval = np.array([f(I, conf) for I in P])
    ga_stats = { "min" : []}
    for i in range(conf.MAX_ITER):
        print(f"{i + 1}/{conf.MAX_ITER}")
        new_P = list(P[P_eval.argsort()][:conf.elit_num])
        for j in range(int((conf.P_SIZE - conf.elit_num)/2)):
            p1, p2 = select(P, P_eval)
            c1, c2 = crossover(p1, p2)
            c1 = mutate(c1, conf)
            c2 = mutate(c2, conf)
            new_P += [c1, c2]
        if (conf.P_SIZE - conf.elit_num) % 2 != 0 : new_P.append(P[-1].copy())
        P = np.array(new_P)
        P_eval = np.array([f(I, conf) for I in P])
        ga_stats["min"].append(P_eval.min())

    return P[P_eval.argmin()], P, ga_stats
    

def main():
    conf = lambda : 0
    #
    conf.MAX_ITER = 100
    conf.P_SIZE = 4
    conf.elit_num = int(0.3*conf.P_SIZE)
    #
    conf.vars = ["x"]
    conf.consts = [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    conf.ops = [add, sub, mul, div]
    conf.min_depth = 2
    conf.max_depth = 10
    conf.term_prob = 0.3
    conf.var_prob = 0.5
    points_x = np.linspace(-5, +5, 5)
    points_y = 2*points_x
    conf.points = np.c_[points_x, points_y].T #np.array([[-2, -1, 0, 1, 2], [4, 1, 0, 1, 4]])

    I_star, P_star, ga_stats = ga(conf)
    print(f"I* : {I_star}")
    print(f"eval(I*) : {ga_stats['min'][-1]}")

    fig, (ax1, ax2) = plt.subplots(1, 2)
    #
    fig.tight_layout()
    ax1.plot(ga_stats["min"])
    ax1.set_xlabel("iter")
    ax1.set_ylabel("eval")
    #
    xs = np.linspace(min(conf.points[0]), max(conf.points[0]), 100)
    ys = [I_star.compute(x=xx) for xx in xs]
    f_star_plt, = ax2.plot(xs, ys)
    ax2.scatter(conf.points[0], conf.points[1], facecolors="none", edgecolors=f_star_plt.get_color())
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.axis("scaled")
    plt.show()

if __name__ == "__main__":
    main()
