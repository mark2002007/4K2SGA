import time
import numpy as np
import matplotlib.pyplot as plt
point = plt.plot(0, 0, "g^")[0]
plt.ylim(0, 5)
plt.xlim(0, 5)
plt.ion()
plt.show()

start_time = time.time()
t = 0
while t < 4:
    end_time = time.time()
    t = end_time - start_time
    print(t)
    point.set_data(t, t)
    plt.pause(1e-10)
