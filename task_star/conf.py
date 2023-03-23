conf = lambda: 0
conf.dim_num = 2
conf.f = lambda x: np.sum((x - np.arange(conf.dim_num) + 1)**2)
conf.a, conf.b = -conf.dim_num, conf.dim_num
conf.MAX_ITER, conf.MAX_ITER_NO_CHANGE = (500,)*2
conf.I_SIZE, conf.P_SIZE = conf.dim_num, 100
conf.elit_num = int(0.3*conf.P_SIZE)
conf.q = 1.5/conf.P_SIZE #linear
#conf.q = 0.5 #non_linear
conf.SELECT_MODEL_TYPE = "linear"

