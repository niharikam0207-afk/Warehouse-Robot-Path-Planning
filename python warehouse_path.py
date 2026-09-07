import heapq
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Warehouse grid representation
# 0 = free cell, 1 = obstacle
warehouse = [
    [0, 0, 0, 0, 1],
    [1, 0, 1, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 1, 0, 1, 0],
    [0, 0, 0, 0, 0]
]

start = (0, 0)   # robot start position
goal = (4, 4)    # target position

# Directions: up, down, left, right
directions = [(0,1), (0,-1), (1,0), (-1,0)]

# ---------------- BFS Implementation ----------------
def bfs(start, goal, grid):
    queue = deque([start])
    visited = set([start])
    parent = {start: None}

    while queue:
        current = queue.popleft()
        if current == goal:
            break
        for dx, dy in directions:
            nx, ny = current[0] + dx, current[1] + dy
            if 0 <= nx < len(grid) and 0 <= ny < len(grid[0]) and grid[nx][ny] == 0:
                if (nx, ny) not in visited:
                    visited.add((nx, ny))
                    parent[(nx, ny)] = current
                    queue.append((nx, ny))

    # reconstruct path
    path = []
    node = goal
    while node:
        path.append(node)
        node = parent.get(node)
    return path[::-1]

# ---------------- A* Implementation ----------------
def heuristic(a, b):
    # Manhattan distance
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def astar(start, goal, grid):
    open_list = []
    heapq.heappush(open_list, (0, start))
    g_score = {start: 0}
    parent = {start: None}

    while open_list:
        _, current = heapq.heappop(open_list)
        if current == goal:
            break
        for dx, dy in directions:
            nx, ny = current[0] + dx, current[1] + dy
            neighbor = (nx, ny)
            if 0 <= nx < len(grid) and 0 <= ny < len(grid[0]) and grid[nx][ny] == 0:
                tentative_g = g_score[current] + 1
                if tentative_g < g_score.get(neighbor, float('inf')):
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_list, (f_score, neighbor))
                    parent[neighbor] = current

    # reconstruct path
    path = []
    node = goal
    while node:
        path.append(node)
        node = parent.get(node)
    return path[::-1]

# ---------------- Run Both Algorithms ----------------
bfs_path = bfs(start, goal, warehouse)
astar_path = astar(start, goal, warehouse)

print("BFS Path:", bfs_path)
print("A* Path:", astar_path)

# ---------------- Visualization ----------------
fig, ax = plt.subplots()
ax.set_xticks(range(len(warehouse[0])))
ax.set_yticks(range(len(warehouse)))
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.grid(True)

# Draw obstacles
for i in range(len(warehouse)):
    for j in range(len(warehouse[0])):
        if warehouse[i][j] == 1:
            ax.add_patch(plt.Rectangle((j,i),1,1,color='black'))

# Robot marker (red circle)
robot, = ax.plot([], [], 'ro', markersize=12)

def init():
    # Start robot at initial position (must be lists, not floats)
    robot.set_data([start[1]+0.5], [start[0]+0.5])
    return robot,

def animate(k):
    # Move robot along the A* path
    x, y = astar_path[k][1], astar_path[k][0]
    robot.set_data([x+0.5], [y+0.5])
    return robot,

ani = animation.FuncAnimation(fig, animate, frames=len(astar_path),
                              init_func=init, interval=500, blit=True)

plt.show()
