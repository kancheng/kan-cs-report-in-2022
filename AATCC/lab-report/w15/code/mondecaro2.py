# 求解定积分 X^2 区间 [1， 2]; 平均法
import numpy as np
import tqdm
sum = 0
for i in tqdm.tqdm(range(3000000)):
        random_x = np.random.uniform(1, 2, size=None)
        # None 是默认的也可以不写
        a =  np.square(random_x)
        sum += a*(2-1)
area = sum/3000000
print('calculate by mean_average:{}'.format(area))
