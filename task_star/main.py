from lib_ga import *
from conf import *
import matplotlib.pyplot as plt

def main():
    Gamma_1 = lambda 




    P, P_eval_stats = ga(conf)
    print(f"f(x*) = {P_eval_stats['min'][-1]}")    
    fig, (ax1, ax2) = plt.subplots(1, 2)
    
    ax1.semilogy(P_eval_stats["min"])
    ax1.semilogy(P_eval_stats["max"])
    ax1.semilogy(P_eval_stats["mean"])
    
    ax2.plot(delta_history)
    plt.show()
    
if __name__ == "__main__":
    main()

