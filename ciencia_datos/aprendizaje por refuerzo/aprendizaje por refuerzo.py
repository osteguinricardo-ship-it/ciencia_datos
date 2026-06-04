# ==========================================
# IMPORTAR LIBRERÍAS
# ==========================================

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# ==========================================
# DEFINIR EL LABERINTO
# ==========================================

maze = np.array([
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    [1, 1, 1, 0, 1, 0, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 1, 0, 0, 1],
    [1, 0, 1, 1, 1, 1, 1, 0, 1, 1],
    [1, 0, 1, 0, 0, 0, 0, 0, 1, 1],
    [1, 0, 1, 0, 1, 1, 1, 0, 1, 1],
    [1, 0, 1, 0, 1, 0, 0, 0, 1, 1],
    [1, 0, 1, 0, 1, 0, 1, 0, 0, 1],
    [1, 1, 1, 0, 1, 1, 1, 1, 0, 0]
])

start = (0, 0)
goal = (9, 9)

# ==========================================
# PARÁMETROS DE REINFORCEMENT LEARNING
# ==========================================

num_episodes = 5000
alpha = 0.1
gamma = 0.9
epsilon = 0.5

reward_fire = -10
reward_goal = 50
reward_step = -1

# Movimientos:
# izquierda, derecha, arriba, abajo
actions = [
    (0, -1),
    (0, 1),
    (-1, 0),
    (1, 0)
]

# Tabla Q
Q = np.zeros(maze.shape + (len(actions),))

# ==========================================
# FUNCIÓN PARA VALIDAR MOVIMIENTOS
# ==========================================

def is_valid(pos):
    r, c = pos

    if r < 0 or r >= maze.shape[0]:
        return False

    if c < 0 or c >= maze.shape[1]:
        return False

    if maze[r, c] == 1:
        return False

    return True

# ==========================================
# ELEGIR ACCIÓN
# ==========================================

def choose_action(state):

    if np.random.random() < epsilon:
        return np.random.randint(len(actions))
    else:
        return np.argmax(Q[state])

# ==========================================
# ENTRENAMIENTO Q-LEARNING
# ==========================================

rewards_all_episodes = []

for episode in range(num_episodes):

    state = start
    total_rewards = 0
    done = False

    while not done:

        action_index = choose_action(state)
        action = actions[action_index]

        next_state = (
            state[0] + action[0],
            state[1] + action[1]
        )

        # Recompensas
        if not is_valid(next_state):

            reward = reward_fire
            done = True

        elif next_state == goal:

            reward = reward_goal
            done = True

        else:

            reward = reward_step

        # Actualizar Q-table
        old_value = Q[state][action_index]

        next_max = (
            np.max(Q[next_state])
            if is_valid(next_state)
            else 0
        )

        Q[state][action_index] = old_value + alpha * (
            reward + gamma * next_max - old_value
        )

        state = next_state
        total_rewards += reward

    epsilon = max(0.01, epsilon * 0.995)

    rewards_all_episodes.append(total_rewards)

# ==========================================
# OBTENER LA MEJOR RUTA
# ==========================================

def get_optimal_path(Q, start, goal, actions, maze, max_steps=200):

    path = [start]
    state = start
    visited = set()

    for _ in range(max_steps):

        if state == goal:
            break

        visited.add(state)

        best_action = None
        best_value = -float('inf')

        for idx, move in enumerate(actions):

            next_state = (
                state[0] + move[0],
                state[1] + move[1]
            )

            if (
                0 <= next_state[0] < maze.shape[0]
                and
                0 <= next_state[1] < maze.shape[1]
                and
                maze[next_state] == 0
                and
                next_state not in visited
            ):

                if Q[state][idx] > best_value:

                    best_value = Q[state][idx]
                    best_action = idx

        if best_action is None:
            break

        move = actions[best_action]

        state = (
            state[0] + move[0],
            state[1] + move[1]
        )

        path.append(state)

    return path

optimal_path = get_optimal_path(
    Q,
    start,
    goal,
    actions,
    maze
)

# ==========================================
# GRAFICAR LABERINTO Y RUTA
# ==========================================

def plot_maze_with_path(path):

    cmap = ListedColormap(['#eef8ea', '#a8c79c'])

    plt.figure(figsize=(8, 8))
    plt.imshow(maze, cmap=cmap)

    plt.scatter(
        start[1],
        start[0],
        marker='o',
        color='green',
        edgecolors='black',
        s=200,
        label='Inicio'
    )

    plt.scatter(
        goal[1],
        goal[0],
        marker='*',
        color='darkgreen',
        edgecolors='black',
        s=300,
        label='Meta'
    )

    rows, cols = zip(*path)

    plt.plot(
        cols,
        rows,
        linewidth=4,
        label='Ruta aprendida'
    )

    plt.title('Reinforcement Learning - Laberinto')

    plt.gca().invert_yaxis()

    plt.xticks(range(maze.shape[1]))
    plt.yticks(range(maze.shape[0]))

    plt.grid(True)
    plt.legend()

    plt.show()

plot_maze_with_path(optimal_path)

# ==========================================
# GRAFICAR RECOMPENSAS
# ==========================================

def plot_rewards(rewards):

    plt.figure(figsize=(10, 5))

    plt.plot(rewards)

    plt.title('Recompensas por episodio')
    plt.xlabel('Episodio')
    plt.ylabel('Recompensa')

    plt.grid(True)

    plt.show()

plot_rewards(rewards_all_episodes)