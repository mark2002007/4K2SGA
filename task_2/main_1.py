import numpy as np
import matplotlib.pyplot as plt
import time

def select(P, P_eval, q, model_type):
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
    
def crossover(p1, p2, p = 0.8):
    if np.random.rand() < p:
        mask = np.random.rand(len(p1))
        c1 = mask*p1 + (1-mask)*p2
        c2 = mask*p2 + (1-mask)*p1
        return c1, c2
    return p1, p2

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

delta_history = []
def mutate(I, a, b, t, T, beta = 5, p = 0.8): #beta \in [3, 5]
    if np.random.rand() < p:
        #delta = (1 - np.random.rand()**((1-t/T)*beta))*(b-a)
        scale, loc = 15, 0
        delta = (1 - sigmoid(t/T*2*scale - scale - loc))*(b-a)
        delta_history.append(delta)                
        #
        mp = np.random.randint(len(I))
        lb = np.maximum(a, I[mp] - delta)
        ub = np.minimum(b, I[mp] + delta)
        new_I = I.copy()
        new_I[mp] = np.random.uniform(lb, ub)
        return new_I
    return I.copy()

def ga(conf):
    #
    P = np.random.uniform(conf.a, conf.b, size=(conf.P_SIZE, conf.I_SIZE))
    P_eval = np.apply_along_axis(conf.f, 1, P).flatten()
    pem, inc = np.inf, 0 #P_eval_min, iter_no_change
    P_eval_stats = {"max" : [], "min" : [], "mean" : []}
    for i in range(conf.MAX_ITER):
        print(f"{i + 1}/{conf.MAX_ITER}")
        new_P = list(P[P_eval.argsort()][:conf.elit_num])
        for j in range(int((conf.P_SIZE-conf.elit_num)/2)):
            p1, p2 = select(P, P_eval, conf.q, conf.SELECT_MODEL_TYPE)
            c1, c2 = crossover(p1, p2)
            new_P += [mutate(I, conf.a, conf.b, i, conf.MAX_ITER) for I in [c1, c2]]
        if (conf.P_SIZE - conf.elit_num) % 2 != 0 : 
            new_P.append(P[-1])
    
        P = np.array(new_P)
        P_eval = np.apply_along_axis(conf.f, 1, P).flatten()
    
        P_eval_stats["min"].append(P_eval.min())
        P_eval_stats["max"].append(P_eval.max())
        P_eval_stats["mean"].append(P_eval.mean())
        
        npem = min(P_eval) #new_P_eval_min
        pem, inc = (pem, inc+1) if npem == pem else (npem, 0)
        if conf.MAX_ITER_NO_CHANGE < inc: break
        
    return P_eval_stats

def main():
    #CONFIG
    conf = lambda: 0
    conf.dim_num = 1
    conf.f = lambda x: x*np.sin(10*np.pi*x) + 1
    #conf.f = lambda x: np.sum((x - np.arange(conf.dim_num) + 1)**2)
    conf.a, conf.b = -1, 2
    conf.MAX_ITER, conf.MAX_ITER_NO_CHANGE = (500,)*2
    conf.I_SIZE, conf.P_SIZE = conf.dim_num, 63
    conf.elit_num = int(0.3*conf.P_SIZE)
    conf.q = 1.5/conf.P_SIZE #linear
    #conf.q = 0.5 #non_linear
    conf.SELECT_MODEL_TYPE = "linear"

    P_eval_stats = ga(conf)
    print(f"f(x*) = {P_eval_stats['min'][-1]}")    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    
    ax1.plot(P_eval_stats["min"])
    #ax1.plot(P_eval_stats["max"])
    #ax1.plot(P_eval_stats["mean"])
    
    ax2.plot(delta_history)

    xs = np.linspace(conf.a, conf.b, 1000)
    ys = np.vectorize(conf.f)(xs)
    ax3.plot(xs, ys)
    
    plt.show()
if __name__ == "__main__":
    main()

