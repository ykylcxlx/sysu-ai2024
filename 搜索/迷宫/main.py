from collections import deque
with open("maze.txt",'r',encoding='gbk') as f:
    data = f.read().splitlines()
    f.close()
row = len(data)
col = len(data[0])
direction =  [(0,1),(1,0),(0,-1),(-1,0)]
path = []
visited = [[0 for i in range(col)] for j in range(row)]
def notin(i,j):
    return i < 0 or j < 0 or i >= row or j >= col
    #print(data)
def dfs():
    for i in range(row):
        for j in range(col):
            if data[i][j] == 'S':
                dfsTraverse(i,j)
    return path

def dfsTraverse(i,j):
    visited[i][j] = 1
    print(i,j)
    #print(path)
    if data[i][j] =='E':
        print("end")
        return True
    else:
        for d in direction:
            nexti = i + d[0]
            nextj = j + d[1]
            # if nexti < 0 or nextj < 0 or nexti >= row or nextj >= col:
            #     continue
            if not notin(nexti,nextj) and (data[nexti][nextj] == '0' or data[nexti][nextj] == 'E') and visited[nexti][nextj] == 0:
                #visited[nexti][nextj] = 1
                path.append((nexti,nextj))
                if dfsTraverse(nexti,nextj):
                    return True
        # if path:
        path.pop()
        #     print(path)
        return False
def bfs():
    start = ()
    for i in range(row):
        for j in range(col):
            if data[i][j] == 'S':
                start = (i,j)
                break
    queue = deque([start])
    all_path = deque([start])
    #path = []

    while queue:
        visited[i][j] = 1
        i,j = queue.popleft()
        cur_path = list(all_path.popleft())
        if data[i][j] == 'E':
            #path = cur_path
            break

        for d in direction:
                nexti = i + d[0]
                nextj = j + d[1]
                # if nexti < 0 or nextj < 0 or nexti >= row or nextj >= col:
                #     continue
                if not notin(nexti,nextj) and (data[nexti][nextj] == '0' or data[nexti][nextj] == 'E') and visited[nexti][nextj] == 0:
                    #visited[nexti][nextj] = 1
                    queue.append((nexti,nextj))
                    all_path.append(cur_path+[(nexti,nextj)])
                    #path.append((nexti,nextj))
    print(cur_path)
    return cur_path
    


from queue import PriorityQueue

def manhattan_distance(point1, point2):
    # 计算曼哈顿距离
    return abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])

def a_star():
    start = ()
    end = ()
    for i in range(row):
        for j in range(col):
            if data[i][j] == 'S':
                start = (i, j)
            elif data[i][j] == 'E':
                end = (i, j)

    # 使用优先级队列按估价函数值进行排序
    queue = PriorityQueue()
    queue.put((0, start))  # 初始节点，代价为0
    parent = {}  # 记录每个位置的父节点
    g_score = {start: 0}  # 记录起点到每个位置的代价
    f_score = {start: manhattan_distance(start, end)}  # 记录起点经过每个位置到终点的估价函数值

    while not queue.empty():
        current = queue.get()[1]  # 获取当前位置

        if current == end:
            # 找到终点，回溯路径
            path = []
            while current in parent:
                path.append(current)
                current = parent[current]
            path.append(start)
            path.reverse()
            return path

        for d in direction:
            nexti = current[0] + d[0]
            nextj = current[1] + d[1]
            if nexti < 0 or nextj < 0 or nexti >= row or nextj >= col:
                continue
            if data[nexti][nextj] == '0':
                new_g_score = g_score[current] + 1  # 到达下一个位置的代价加1
                if (nexti, nextj) not in g_score or new_g_score < g_score[(nexti, nextj)]:
                    # 更新到达下一个位置的代价和估价函数值
                    g_score[(nexti, nextj)] = new_g_score
                    f_score[(nexti, nextj)] = new_g_score + manhattan_distance((nexti, nextj), end)
                    parent[(nexti, nextj)] = current
                    queue.put((f_score[(nexti, nextj)], (nexti, nextj)))

    return None

result = a_star()
print(result)

#dfs()
#bfs()
a_star()