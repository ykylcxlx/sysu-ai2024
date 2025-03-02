from collections import deque

# 定义船的容量和初始状态
MAX_CAPACITY = 4
INITIAL_STATE = (3, 3, 1)  # (传教士数量, 野人数量, 船的位置(1表示在起始岸))

def is_valid(state):
    """检查状态是否合法"""
    missionaries, cannibals, boat = state

    # 传教士和野人数量不能为负数
    if missionaries < 0 or cannibals < 0:
        return False

    # 传教士的数量不能小于野人的数量（在某一岸上）
    if missionaries > 0 and missionaries < cannibals:
        return False

    # 船上的人数不能超过船的容量
    if missionaries + cannibals > MAX_CAPACITY:
        return False

    return True

def successors(state):
    """生成所有可能的合法状态"""
    missionaries, cannibals, boat = state
    result = []

    # 船在起始岸
    if boat == 1:
        for m in range(MAX_CAPACITY + 1):
            for c in range(MAX_CAPACITY - m + 1):
                if (m + c) > 0 and (m + c) <= MAX_CAPACITY and (m>c or m==0 and c>0):
                    result.append((missionaries - m, cannibals - c, 0))
    # 船在对岸
    else:
        for m in range(MAX_CAPACITY + 1):
            for c in range(MAX_CAPACITY - m + 1):
                if (m + c) > 0 and (m + c) <= MAX_CAPACITY and (m>c or m==0 and c>0):
                    result.append((missionaries + m, cannibals + c, 1))
    return result

def bfs_with_cycle_detection():
    """带环检测的宽度优先搜索"""
    visited = set()
    queue = deque([(INITIAL_STATE, [])])  # (state, path)

    while queue:
        state, path = queue.popleft()
        print(path)
        if state == (0, 0, 0):  # 目标状态
            return path + [state]

        visited.add(state)

        for succ in successors(state):
            if is_valid(succ) and succ not in visited:
                queue.append((succ, path + [state]))

    return None

def print_solution(solution):
    """打印解决方案"""
    if solution is None:
        print("无解")
    else:
        print("解决方案：")
        for i, state in enumerate(solution):
            missionaries, cannibals, boat = state
            side = "起始岸" if boat == 1 else "对岸"
            print(f"步骤 {i+1}: 传教士: {missionaries}, 野蛮人: {cannibals}, 船在{side}")

if __name__ == "__main__":
    solution = bfs_with_cycle_detection()
    print_solution(solution)
