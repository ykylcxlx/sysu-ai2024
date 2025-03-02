with open("maze.txt", 'r', encoding='gbk') as f:
    data = f.read().splitlines()
    row = len(data)
    col = len(data[0])
    f.close()

direction = [(0, 1), (1, 0), (0, -1), (-1, 0)]
path = []
visited = [[0 for _ in range(col)] for _ in range(row)]


def dfs():
    for i in range(row):
        for j in range(col):
            if data[i][j] == 'S':
                dfsTraverse(i, j)
    return path

def dfsTraverse(i, j):
    # if row < 0 or i >= row or i < 0 or j >= col or data[i][j] == '1' or visited[i][j]:
    #         return False
    visited[i][j] = 1

    if data[i][j] == 'E':
        return True
    else:
        for d in direction:
            nexti = i + d[0]
            nextj = j + d[1]
            # if nexti < 0 or nextj < 0 or nexti >= row or nextj >= col:
            #     continue
            if isin(nexti,nextj) and data[nexti][nextj] == '0' and visited[nexti][nextj] == 0:
                path.append((nexti, nextj))
                if dfsTraverse(nexti, nextj):
                    return True
                path.pop()  # 在回溯时移除路径上的最后一个位置信息

    return False
def dfs_helper(row, col):
    if row < 0 or row >= rows or col < 0 or col >= cols or puzzle[row][col] == '1' or visited[row][col]:
        return False

    visited[row][col] = True
    path.append((row, col))

    if (row, col) == end:
        return True

    for dx, dy in directions:
        if dfs_helper(row + dx, col + dy):
            return True

    path.pop()  # 如果该位置无法到达终点，则将其从路径中移除
    return False


result = dfs()
print(result)