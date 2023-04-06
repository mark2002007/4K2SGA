#from lib_ga import *
#from conf import *
import time
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(linewidth=np.inf)
circle = lambda t: np.array([np.cos(t), np.sin(t)])

#Helpers
def is_in_circle(point, r): 
    return np.linalg.norm(point) < r

def is_in_area(point, r_1, r_2):
    return is_in_circle(point, r_1) and not is_in_circle(point, r_2)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def vec_to_points(vec):
    return vec.reshape((-1, 2))  

def distance(point1, point2):
    return np.linalg.norm(point1 - point2)

def distances(point, points):
    return np.array([distance(point, p) for p in points])

def is_close_to_any(point, points, offset):
    return any(distances(point, points) < offset)

def points_inside(points, r):
    return sum([is_in_circle(p, r) for p in points])

def points_inside_percent(points, r):
    return points_inside(points, r)/len(points)
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
    r_1, r_2, p, area_offset, points_offset, patience = conf.r_1, conf.r_2, conf.co_prob, conf.area_offset, conf.points_offset, conf.patience
    if np.random.rand() < p:
        mask = np.random.rand(len(p_1))
        c_1 = mask*p_1 + (1-mask)*p_2
        c_2 = mask*p_2 + (1-mask)*p_1
        for i in range(0, len(p_1), 2):
            cnt = 0
            while is_in_area(c_1[i:i+2], r_1 + area_offset, r_2) or \
                  is_in_area(c_2[i:i+2], r_1 + area_offset, r_2) or \
                  is_close_to_any(c_1[i:i+2], vec_to_points(np.r_[c_1[:i], c_1[i+2:]]), points_offset) or \
                  is_close_to_any(c_2[i:i+2], vec_to_points(np.r_[c_2[:i], c_2[i+2:]]), points_offset) or \
                  points_inside(vec_to_points(c_1), r_2) < 1 or \
                  points_inside(vec_to_points(c_2), r_2) < 1: 
#                input("crossover_repeat")
                mask = np.random.rand(2)
                c_1[i:i+2] = mask*p_1[i:i+2] + (1-mask)*p_2[i:i+2]
                c_2[i:i+2] = mask*p_2[i:i+2] + (1-mask)*p_1[i:i+2]
                cnt+=1
                if cnt == patience: return p_1, p_2
        return c_1, c_2
    return p_1, p_2


def mutate_sigmoid(I, t, conf):
    T, a, b, scale, loc, r_1, r_2, area_offset, points_offset, patience, p = conf.max_iter, conf.a, conf.b, conf.mut_scale, conf.mut_loc, conf.r_1, conf.r_2, conf.area_offset, conf.points_offset, conf.patience, conf.mut_prob
    if np.random.rand() < p:
        delta = (1 - sigmoid(t/T*2*scale - scale - loc))*(b - a)
        mp = np.random.randint(len(I))
        lb = max(I[mp] - delta, a)
        ub = min(I[mp] + delta, b)
        new_I = I.copy()
        while True:
            new_I[mp] = np.random.uniform(lb, ub)
            cnt = 0
            if not is_in_area     (new_I[mp//2*2:mp//2*2+2], r_1 + area_offset, r_2) and \
               not is_close_to_any(new_I[mp//2*2:mp//2*2+2], vec_to_points(np.r_[new_I[:mp//2*2], new_I[mp//2*2+2:]]), points_offset) and \
               not points_inside(vec_to_points(new_I), r_2) < 1: \
               break
            cnt += 1
            if cnt == patience: return I.copy(), None
        return new_I, delta
    return I.copy(), None

#MFS
def MFS(y_, x_1, x_2, f_1, f_2):
    n = len(y_)
    Phi = lambda x, y: 1/(2*np.pi)*np.log(1/np.linalg.norm(x - y)) 
    that_ = 4*np.pi/n * np.arange(1, int(n/2) + 1)
    xhat_1_, xhat_2_ = np.array([x_1(t) for t in that_]), np.array([x_2(t) for t in that_])
   
#    fig, ax = plt.subplots()
#    ax.scatter(xhat_1_[:,0], xhat_1_[:,1])
#    ax.scatter(xhat_2_[:,0], xhat_2_[:,1])
#    plt.show()
#    input()

    A = np.array([[Phi(x, y) for y in y_] for x in np.r_[xhat_1_, xhat_2_]])
    b = np.array([f_1(x) for x in xhat_1_] + [f_2(x) for x in xhat_2_])
    l = np.linalg.solve(A, b)
    u_n = lambda x : np.sum(l*Phi(x, y_))
    return u_n


def f(I, x_1, x_2, f_1, f_2):
    y_ = vec_to_points(I)
    u_n = MFS(y_, x_1, x_2, f_1, f_2)
#    t_ = np.linspace(0, 2*np.pi, 7)[:-1] 
    n = len(y_)
    t_ = 4*np.pi/n * np.arange(1, int(n/2) + 1)

    x_1_ = [x_1(t) for t in t_]
    x_2_ = [x_2(t) for t in t_]
    u_n_1_ = np.array([u_n(x) for x in x_1_])
    u_n_2_ = np.array([u_n(x) for x in x_2_])
    f_1_ = np.array([f_1(x) for x in x_1_])
    print(f"u_n_1 : \n{u_n_1_}")
    print(f"f_1_ : \n{f_1_}")
    f_2_ = np.array([f_2(x) for x in x_2_])
    return np.sqrt(np.linalg.norm(u_n_1_ - f_1_)**2 + np.linalg.norm(u_n_2_ - f_2_)**2)
    
#
def main():
    conf = lambda: 0
    #Config : General
    conf.max_iter, conf.max_iter_no_change = (50,)*2
    conf.print_interval = 1
    conf.p_size = 9
    conf.a = -15
    conf.b = 15
    conf.elit_num = int(0.2*conf.p_size)
    conf.patience = 500
    #Config : Select
    conf.sel_model_type = "linear"
    conf.sel_q = 1.5/conf.p_size if conf.sel_model_type == "linear" else 0.5
    #Config : Crossover
    conf.co_prob = 1
    #Config : Mutate Sigmoid 
    conf.mut_prob = 1
    conf.mut_scale = 15
    conf.mut_loc = 0
    #Config : Problem
    conf.n = 30 #must be even
    conf.r_1 = 3
    conf.r_2 = 2
    conf.x_1 = lambda t: conf.r_1*circle(t)
    conf.x_2 = lambda t: conf.r_2*circle(t)
    conf.f_1 = lambda x: x[0] + x[1]
    conf.f_2 = lambda x: x[0] + x[1]
    conf.area_offset = 0.01
    conf.points_offset = 0.01

    #Generate Population
    t_ = np.linspace(0, 2*np.pi-1e-1, conf.p_size + 1)[:-1]
    points_inside_n = 1 #int(min((conf.r_2 - conf.area_offset - conf.points_offset)/conf.points_offset, conf.n/2))
    c2 = np.linspace(conf.r_2 - conf.area_offset, conf.points_offset, points_inside_n + 2         )[1:-1][::-1]
    c1 = np.linspace(conf.r_1 + conf.area_offset, conf.b            , conf.n - points_inside_n + 2)[1:-1]
    c_ = np.r_[c2, c1]
    P = np.array([np.array([c*circle(t) for c in c_]).flatten() for t in t_])
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
    plotted_points_ = []
    for I in P:
        plotted_points_.append(ax.scatter(*list(zip(*vec_to_points(I)))))
    input() 
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
            for I, plotted_points in zip(P, plotted_points_):
                plotted_points.set_offsets(np.c_[*list(zip(*vec_to_points(I)))])
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
    ax1.set_xlabel("iteration")
    ax1.set_ylabel("eval")
    ax2.plot(ga_stats["delta"])
    input()
    
if __name__ == "__main__":
    main()
    pass
