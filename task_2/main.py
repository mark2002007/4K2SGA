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

def mutate(I, a, b, t, T, beta = 3, p = 0.8): #beta \in [3, 5]
    if np.random.rand() < p:
        delta = (1 - np.random.rand()**((1-t/T)*beta))*(b-a)
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
    P_eval = np.apply_along_axis(conf.f, 1, P)
    pem, inc = np.inf, 0 #P_eval_min, iter_no_change
    P_eval_stats = {"max" : [], "min" : [], "mean" : []}
    for i in range(conf.MAX_ITER):
        #print(f"{i + 1}/{conf.MAX_ITER}")
        new_P = list(P[P_eval.argsort()][:conf.elit_num])
        for j in range(int((conf.P_SIZE-conf.elit_num)/2)):
            p1, p2 = select(P, P_eval, conf.q, conf.SELECT_MODEL_TYPE)
            c1, c2 = crossover(p1, p2)
            new_P += [mutate(I, conf.a, conf.b, i, conf.MAX_ITER) for I in [c1, c2]]
        if (conf.P_SIZE - conf.elit_num) % 2 != 0 : new_P += P[-1]
        P = np.array(new_P)

        P_eval = np.apply_along_axis(conf.f, 1, P)
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
    conf.dim_num = 2
    conf.f = lambda x: np.sum((x - np.arange(conf.dim_num) + 1)**2)
    conf.a, conf.b = -conf.dim_num, conf.dim_num
    conf.MAX_ITER, conf.MAX_ITER_NO_CHANGE = (250,)*2
    conf.I_SIZE, conf.P_SIZE = conf.dim_num, 100
    conf.elit_num = int(0.3*conf.P_SIZE)
    #conf.q = 1.5/conf.P_SIZE
    #conf.q = 0.5
    #conf.SELECT_MODEL_TYPE = "roulette"

    #PLOT1
    fig, ax = plt.subplots()
    for SMT in ["roulette", "linear", "non_linear"]:
        print(SMT)
        conf.SELECT_MODEL_TYPE = SMT 
        conf.q = 1.5/conf.P_SIZE if SMT == "linear" else 0.3
        P_eval_stats = ga(conf)
        ax.semilogy(P_eval_stats["min"], label=SMT)
    print(f"f(x*) = {P_eval_stats['min'][-1]}")
    ax.set_xlabel("iteration")
    ax.set_ylabel("eval")
    ax.legend()
    plt.show()
    
    #PLOT2
#    conf.MAX_ITER = 250
#    conf.SELECT_MODEL_TYPE = "linear"
#    fig, ax = plt.subplots()
#    P_eval_min_ = []
#    q_ = np.linspace(1, 2, 15)/conf.P_SIZE
#    for q in q_:
#        print(f"q = {q}")
#        conf.q = q 
#        P_eval_stats = ga(conf)
#        P_eval_min_.append(P_eval_stats["min"][-1])
#    ax.semilogy(q_, P_eval_min_)
#    ax.set_xlabel("q")
#    ax.set_ylabel("eval")
#    ax.legend()
#    plt.show()
    
if __name__ == "__main__":
    main()

