import numpy as np
import matplotlib.pyplot as plt


def select(P, P_eval, q, model_name = "linear"):
    P_len = len(P)
    P = P[P_eval.argsort()]
    P_eval = P_eval[P_eval.argsort()]
    ranks = np.arange(P_len)
    match model_name:
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

def mutate(I, a, b, t, T, beta = 4, p = 0.8):
    if np.random.rand() < p:
        delta = (1 - np.random.rand()**((1-t/T)*beta))*(b-a)
        mp = np.random.randint(len(I))
        lb = np.maximum(a, I[mp] - delta)
        ub = np.minimum(b, I[mp] + delta)
        new_I = I.copy()
        new_I[mp] = np.random.uniform(lb, ub)
        return new_I
    return I.copy()

def main():
    dim_num = 2
    f = lambda x: np.sum((x - np.arange(dim_num) + 1)**2)
    a, b = -5, 5
    #
    MAX_ITER, MAX_ITER_NO_CHANGE = 2000, 1000
    I_SIZE, P_SIZE = dim_num, 16
    q = 1.5/P_SIZE
    elit_num = int(0.3*P_SIZE)
    #
    P = np.random.uniform(a, b, size=(P_SIZE, I_SIZE))
    P_eval = np.apply_along_axis(f, 1, P)
    pem, inc = np.inf, 0 #P_eval_min, iter_no_change
    P_eval_stats = {"max" : [], "min" : [], "mean" : []}
    for i in range(MAX_ITER):
        print(f"{i}/{MAX_ITER}")
        new_P = list(P[P_eval.argsort()][:elit_num])
        for j in range(int((P_SIZE-elit_num)/2)):
            p1, p2 = select(P, P_eval, q)
            c1, c2 = crossover(p1, p2)
            new_P += [mutate(I, a, b, i, MAX_ITER) for I in [c1, c2]]
        if (P_SIZE-elit_num) % 2 != 0 : new_P += P[-1]
        
        P = np.array(new_P)
        P_eval = np.apply_along_axis(f, 1, P)
        P_eval_stats["min"].append(P_eval.min())
        P_eval_stats["max"].append(P_eval.max())
        P_eval_stats["mean"].append(P_eval.mean())

        npem = min(P_eval) #new_P_eval_min
        pem, inc = (pem, inc+1) if npem == pem else (npem, 0)
        if MAX_ITER_NO_CHANGE < inc: break

    print(f"f(x*) = f({P[P_eval.argmin()]}) = {min(P_eval)}")
    plt.semilogy(P_eval_stats["min"], label="best")
    plt.semilogy(P_eval_stats["max"], label="worst")
    plt.semilogy(P_eval_stats["mean"], label="average")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()

