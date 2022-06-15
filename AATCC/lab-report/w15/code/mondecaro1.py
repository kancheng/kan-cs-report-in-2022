# 求解定积分 x^2 区间[1, 2]; 投点法
import numpy as np
import tqdm
sum = 0
for i in tqdm.tqdm(range(3000000)):

        random_generate = np.array([np.random.uniform(1, 2), np.random.uniform(0, 4)])
        if np.square(random_generate[0]) > random_generate[1]:
                sum += 1
print(sum)
area = 4 * sum / 3000000
print('Area is:{}'.format(area))