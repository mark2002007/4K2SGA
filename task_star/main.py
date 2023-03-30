#from lib_ga import *
#from conf import *
import time
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(linewidth=np.inf)
circle = lambda t: np.array([np.cos(t), np.sin(t)])

#Helpers
def in_circle(point, r):
    return np.linalg.norm(point) < r

def in_area(point, r_1, r_2):
    return in_circle(point, r_1) and not in_circle(point, r_2)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def vec_to_dots(vec):
    return vec.reshape((-1, 2))

#GA
def select_smart_scales(P, P_eval, conf):
    q, model_type = conf.sel_q, conf.sel_model_type
    P_len = len(P)
    P = P[P_eval.argsort()]
    P_eval = P_eval[P_eval.argsort()]
    ranks = np.arange(P_len)
    match model_type:
        case "roulette":
            e = P_eval[-1] - P_eval + 1
            probs = e / np.sum(e)
        case "linear":
            r = 2*(q*P_len - 1)/(P_len*(P_len - 1))
            probs = q - ranks*r
        case "non_linear":
            probs = q*(1 - q)**ranks
        case _: raise Exception("ValueError: unknown selector")
    probs_cum = np.cumsum(probs)
    pointer = np.random.rand(2)
    return P[np.digitize(pointer, probs_cum)]

def crossover_uniform(p_1, p_2, conf):
    r_1, r_2, p, area_offset = conf.r_1, conf.r_2, conf.co_prob, conf.area_offset
    if np.random.rand() < p:
        mask = np.random.rand(len(p_1))
        c_1 = mask*p_1 + (1-mask)*p_2
        c_2 = mask*p_2 + (1-mask)*p_1
        for i in range(0, len(p_1), 2):
            while in_area(c_1[i:i+2], r_1 + area_offset, r_2) or in_area(c_2[i:i+2], r_1 + area_offset, r_2):
               mask = np.random.rand(2)
               c_1[i:i+2] = mask*p_1[i:i+2] + (1-mask)*p_2[i:i+2]
               c_2[i:i+2] = mask*p_2[i:i+2] + (1-mask)*p_1[i:i+2]
        return c_1, c_2
    return p_1, p_2


def mutate_sigmoid(I, t, conf):
    T, a, b, scale, loc, r_1, r_2, area_offset, p = conf.max_iter, conf.a, conf.b, conf.mut_scale, conf.mut_loc, conf.r_1, conf.r_2, conf.area_offset, conf.mut_prob
    if np.random.rand() < p:
        delta = (1 - sigmoid(t/T*2*scale - scale - loc))*(b - a)
        mp = np.random.randint(len(I))
        lb = max(I[mp] - delta, a)
        ub = min(I[mp] + delta, b)
        new_I = I.copy()
        while True:
            new_I[mp] = np.random.uniform(lb, ub)
            if not in_area(new_I[mp//2*2:mp//2*2+2], r_1 + area_offset, r_2): break
        return new_I, delta
    return I.copy(), None

#MFS
def MFS(y_, x_1, x_2, f_1, f_2):
    n = len(y_)
    Phi = lambda x, y: 1/(2*np.pi)*np.log(1/np.linalg.norm(x - y)) 
    that_ = np.arange(1, int(n/2) + 1)
    xhat_1_, xhat_2_ = np.array([x_1(t) for t in that_]), np.array([x_2(t) for t in that_])
    A = np.array([[Phi(x, y) for y in y_] for x in np.r_[xhat_1_, xhat_2_]])
    b = np.array([f_1(x) for x in xhat_1_] + [f_2(x) for x in xhat_2_])
    l = np.linalg.solve(A, b)
    u_n = lambda x : np.sum(l*Phi(x, y_))
    return u_n


def f(I, x_1, x_2, f_1, f_2):
    y_ = vec_to_dots(I)
    u_n = MFS(y_, x_1, x_2, f_1, f_2)
    t_ = np.linspace(0, 2*np.pi, 6)[:-1]
    x_1_ = [x_1(t) for t in t_]
    x_2_ = [x_2(t) for t in t_]
    u_n_1_ = np.array([u_n(x) for x in x_1_])
    u_n_2_ = np.array([u_n(x) for x in x_2_])
    f_1_ = np.array([f_1(x) for x in x_1_])
    f_2_ = np.array([f_2(x) for x in x_2_])
    return np.sqrt(np.linalg.norm(u_n_1_ - f_1_)**2 + np.linalg.norm(u_n_2_ - f_2_)**2)
    
#
def main():
    conf = lambda: 0
    #Config : General
    conf.max_iter, conf.max_iter_no_change = (250,)*2
    conf.print_interval = 1
    conf.p_size = 16
    conf.a = -10
    conf.b = 10
    conf.elit_num = int(0.3*conf.p_size)
    #Config : Select
    conf.sel_model_type= "linear"
    conf.sel_q = 1.5/conf.p_size if conf.sel_model_type == "linear" else 0.5
    #Config : Crossover
    conf.co_prob = 1
    #Config : Mutate Sigmoid 
    conf.mut_prob = 1
    conf.mut_scale = 15
    conf.mut_loc = -10
    #Config : Problem
    conf.n = 2 #must be even
    conf.r_1 = 3
    conf.r_2 = 1.5
    conf.x_1 = lambda t: conf.r_1*circle(t)
    conf.x_2 = lambda t: conf.r_2*circle(t)
    conf.f_1 = lambda x: x[0] + x[1]
    conf.f_2 = lambda x: x[0] * x[1]
    conf.area_offset = 3

    #Generate Population
    t_ = np.linspace(0, 2*np.pi, conf.n + 1)[:-1]
    c_ = np.linspace(conf.r_1 + conf.area_offset, conf.b, conf.p_size + 2)[1:-1]
    P = np.array([np.array([c*circle(t) for t in t_]).flatten() for c in c_])
    P_eval = np.array([f(I, conf.x_1, conf.x_2, conf.f_1, conf.f_2) for I in P])
    
    #Initialize Plotting
    plt.ion()
    fig, ax = plt.subplots()
    pt_ = np.linspace(0, 2*np.pi, 100)
    ax.plot(*list(zip(*[conf.x_1(t) for t in pt_])))
    ax.plot(*list(zip(*[conf.x_2(t) for t in pt_])))
    ax.set_xlim(conf.a, conf.b)
    ax.set_ylim(conf.a, conf.b)
    ax.set_aspect("equal", adjustable="box")
    plotted_points = ax.scatter(*list(zip(*vec_to_dots(P))))
    
    #Iterate
    pem, inc = np.inf, 0 #P_eval_min, iter_no_change
    ga_stats = {"max" : [], "min" : [], "mean" : [], "delta" : []}
    for i in range(conf.max_iter):
        print(f"{i + 1}/{conf.max_iter}")
        #Elit Population
        new_P = list(P[P_eval.argsort()][:conf.elit_num])
        
        #New Population
        for j in range(int((conf.p_size-conf.elit_num)/2)):
            p_1, p_2 = select_smart_scales(P, P_eval, conf)
            c_1, c_2 = crossover_uniform(p_1, p_2, conf)
            c_1, delta_1 = mutate_sigmoid(c_1, i, conf)
            c_2, delta_2 = mutate_sigmoid(c_2, i, conf)
            ga_stats["delta"] += [delta_1, delta_2]
            new_P += [c_1, c_2]
        if (conf.p_size - conf.elit_num) % 2 != 0 : #Add Remaining Individuals 
            new_P.append(P[-1])
        P = np.array(new_P)
        P_eval = np.array([f(I, conf.x_1, conf.x_2, conf.f_1, conf.f_2) for I in P])
        
        #Update Scatter
        if i % conf.print_interval == 0:
            plotted_points.set_offsets(np.c_[*list(zip(*vec_to_dots(P)))])
            input()
        
        #Log
        ga_stats["min"].append(P_eval.min())
        ga_stats["max"].append(P_eval.max())
        ga_stats["mean"].append(P_eval.mean())
        
        #Check Quit Condition
        npem = min(P_eval) #new_P_eval_min
        pem, inc = (pem, inc+1) if npem == pem else (npem, 0)
        if conf.max_iter_no_change < inc: break
    
    #Result
    print(f"P : {P}")
    print(f"P_eval : {P_eval}")
    print(f"f(x*) = {ga_stats['min'][-1]}")   
    input()
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.semilogy(ga_stats["min"])
    ax2.plot(ga_stats["delta"])
    input()
    
if __name__ == "__main__":
    main()
