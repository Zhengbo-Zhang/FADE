matrix = []
n = int(input())
for _ in range(n):
    matrix.append([int(x) for x in input().split()])

dirs = [(-1, 0), (1, 0), (0, 1), (0, -1)]
visited = [[False] * n for _ in range(n)]
visited[0][0] = True

queue = [(0, 0)]
steps = 0
while queue:
    m = len(queue)
    for _ in range(m):
        i, j = queue.pop(0)
        if i == j == n:
            print(steps)
        for dir in dirs:
            x, y = i + dir[0], j + dir[1]
            if 0 <= x < n and 0 <= y < n and matrix[x][y] and not visited[x][y]:
                queue.append((x, y))
        steps += 1
print(-1)
