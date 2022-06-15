import numpy as np
import tqdm
#random_generate = np.random.uniform(low=0.0, high=2.0, size=(1, 1))

#求解 pi
sum = 0
for i in tqdm.tqdm(range(3000000)):
        #random_generate = np.random.rand(2
        random_generate = np.random.uniform(low=0.0, high=2.0, size=(2))
        if np.sum(np.square(random_generate-np.array([1.0, 1.0]))) <=1:
                sum += 1
print(sum)
pi = 4 * (sum / 3000000)
print('pi is:{}'.format(pi))