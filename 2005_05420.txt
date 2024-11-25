import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import heapq

# Global parameters
GRID_SIZE = 20  # Grid size
LOCAL_FOV = 5   # Local field of view radius
NUM_DYNAMIC_OBSTACLES = 100  # Increased number of dynamic obstacles
NUM_ROBOTS = 1
STARTS = [(0, 0)]
GOALS = [(18, 18)]
STATIC_OBSTACLES = set()
while len(STATIC_OBSTACLES) < 40:
    STATIC_OBSTACLES.add((np.random.randint(0, GRID_SIZE), np.random.randint(0, GRID_SIZE)))

# Dynamic obstacles
dynamic_obstacles = [(np.random.randint(0, GRID_SIZE), np.random.randint(0, GRID_SIZE)) for _ in range(NUM_DYNAMIC_OBSTACLES)]
dynamic_directions = [(np.random.choice([-1, 0, 1]), np.random.choice([-1, 0, 1])) for _ in range(NUM_DYNAMIC_OBSTACLES)]


def a_star(start, goal, static_obstacles):
    """Compute the global guidance path using the A* algorithm."""
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            return reconstruct_path(came_from, current)

        for neighbor in get_neighbors(current):
            if neighbor in static_obstacles or not in_bounds(neighbor):
                continue
            tentative_g_score = g_score[current] + 1
            if tentative_g_score < g_score.get(neighbor, float("inf")):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                if neighbor not in [item[1] for item in open_set]:
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
    return []


def heuristic(a, b):
    """Manhattan distance heuristic."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def reconstruct_path(came_from, current):
    """Reconstruct the path from start to goal."""
    path = []
    while current in came_from:
        path.append(current)
        current = came_from[current]
    return path[::-1]


def get_neighbors(pos):
    """Get adjacent cells."""
    x, y = pos
    return [(x + dx, y + dy) for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]]


def in_bounds(pos):
    """Check if position is within grid bounds."""
    x, y = pos
    return 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE


def update_dynamic_obstacles():
    """Move dynamic obstacles."""
    global dynamic_obstacles, dynamic_directions
    for i, (x, y) in enumerate(dynamic_obstacles):
        dx, dy = dynamic_directions[i]
        new_pos = (x + dx, y + dy)
        if new_pos in STATIC_OBSTACLES or not in_bounds(new_pos):
            dynamic_directions[i] = (-dx, -dy)
        else:
            dynamic_obstacles[i] = new_pos


def rl_local_planner(local_fov, global_guidance, robot_pos):
    """Simplified RL-based decision-making."""
    # Get neighbors within the local field of view
    neighbors = get_neighbors(robot_pos)
    valid_moves = [n for n in neighbors if in_bounds(n) and n not in STATIC_OBSTACLES and n not in dynamic_obstacles]

    if not valid_moves:
        return robot_pos  # No valid moves, stay in place

    # Prioritize moves that align with the global guidance
    valid_moves.sort(key=lambda n: heuristic(n, global_guidance[0]))
    return valid_moves[0]  # Choose the best move


# Initialize the global guidance path
global_paths = [a_star(STARTS[0], GOALS[0], STATIC_OBSTACLES)]

# Animation setup
fig, ax = plt.subplots(figsize=(8, 8))
robot_positions = STARTS

def animate(frame):
    global robot_positions, global_paths
    update_dynamic_obstacles()

    # Update robot positions based on local RL planner
    for i in range(NUM_ROBOTS):
        local_fov = [(robot_positions[i][0] + dx, robot_positions[i][1] + dy) for dx in range(-LOCAL_FOV, LOCAL_FOV+1) for dy in range(-LOCAL_FOV, LOCAL_FOV+1)]
        robot_positions[i] = rl_local_planner(local_fov, global_paths[i], robot_positions[i])

    # Remove visited nodes from the global path
    for i in range(NUM_ROBOTS):
        if global_paths[i] and robot_positions[i] == global_paths[i][0]:
            global_paths[i].pop(0)

    # Visualization
    ax.clear()
    ax.set_xlim(-1, GRID_SIZE)
    ax.set_ylim(-1, GRID_SIZE)
    ax.set_title(f"Step {frame}")
    
    # Draw grid
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            if (x, y) in STATIC_OBSTACLES:
                ax.add_patch(plt.Rectangle((x, y), 1, 1, color="gray"))

    # Draw dynamic obstacles
    for x, y in dynamic_obstacles:
        ax.add_patch(plt.Circle((x + 0.5, y + 0.5), 0.3, color="blue"))

    # Draw robots
    for pos in robot_positions:
        ax.add_patch(plt.Circle((pos[0] + 0.5, pos[1] + 0.5), 0.4, color="red"))

    # Draw global paths
    for path in global_paths:
        for x, y in path:
            ax.add_patch(plt.Rectangle((x, y), 1, 1, edgecolor="yellow", fill=False))


ani = FuncAnimation(fig, animate, frames=50, interval=500)
plt.show()

