

nodes = ('A', 'B', 'C', 'G')
distances = {
    'A': {'B': 2, 'C': 4},
    'B': {'A': 1, 'G': 1},
    'C': {'G': 1, 'B': -3},
    'G': {'G': 0}}


unvisited = {node: None for node in nodes} #把None作为无穷大使用
visited = {}   #用来记录已经松弛过的数组
current = 'A'  #要找B点到其他点的距离
currentDistance = 0
unvisited[current] = currentDistance  #B到B的距离记为0

while True:
    print(current)
    for neighbour, distance in distances[current].items():
        if neighbour not in unvisited: continue   #被访问过了，跳出本次循环
        newDistance = currentDistance + distance  #新的距离
        if unvisited[neighbour] is None or unvisited[neighbour] > newDistance: #如果两个点之间的距离之前是无穷大或者新距离小于原来的距离
            unvisited[neighbour] = newDistance  #更新距离
    visited[current] = currentDistance  #这个点已经松弛过，记录
    del unvisited[current]  #从未访问过的字典中将这个点删除
    if not unvisited: break  #如果所有点都松弛过，跳出此次循环
    candidates = [node for node in unvisited.items() if node[1]]  #找出目前还有哪些点未松弛过
    current, currentDistance = sorted(candidates, key=lambda x: x[1])[0]  #找出目前可以用来松弛的点
    print(visited)
print(visited)
