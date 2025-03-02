def DFS(puzzle):
    rows = len(puzzle)
    cols = len(puzzle[0])
    visited = [[False] * cols for _ in range(rows)]  # 记录已访问的位置
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]  # 右、左、下、上
    start = None  # 起点位置
    end = None  # 终点位置

    # 找到起点和终点的位置
    for i in range(rows):
        for j in range(cols):
            if puzzle[i][j] == 'S':
                start = (i, j)
            elif puzzle[i][j] == 'E':
                end = (i, j)

    path = []  # 记录移动过程中的位置信息

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

    dfs_helper(start[0], start[1])
    return path

puzzle = [
    "111111111111111111111111111111111111",
    "1000000000000000000000000000000000S1",
    "101111111111111111111111101111111101",
    "101100010001000000111111100011000001",
    "101101010101011110111111111011011111",
    "101101010101000000000000011011000001",
    "101101010101010111100111000011111101",
    "101001010100010000110111111110000001",
    "101101010111111110110000000011011111",
    "101101000110000000111111111011000001",
    "100001111110111111100000011011111101",
    "111111000000100000001111011010000001",
    "100000011111101111101000011011011111",
    "101111110000001000000011111011000001",
    "100000000111111011111111111011001101",
    "111111111100000000000000000011111101",
    "1E0000000001111111111111111000000001",
    "111111111111111111111111111111111111"
]

result = DFS(puzzle)
print(result)