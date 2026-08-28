from agentes.base import Agent
import numpy as np
from collections import defaultdict
import pickle

class QAgent(Agent):
    """
    Nota: epsilon se define 0.0 por defecto para que el agente no tome acciones aleatorias en test.
    """
    def __init__(self, actions, game=None, learning_rate=0.1, discount_factor=0.99,
             epsilon=0.0, epsilon_decay=0.995, min_epsilon=0.01,
             load_q_table_path=None):
        """
        Inicializa el agente Q-Learning.
        Si se encuentra un archivo de Q-table entrenada (por defecto 'flappy_birds_q_table_final.pkl'),
        lo carga automáticamente. De lo contrario, crea una tabla vacía.
        """
        super().__init__(actions, game)

        # Hiperparámetros
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

        # Cargar Q-table entrenada (ruta relativa al proyecto)
        import os
        try:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            # Buscar archivo explícito o usar el default en la carpeta superior
            q_table_file = load_q_table_path or os.path.join(base_dir, "..", "flappy_birds_q_table_final.pkl")
            q_table_file = os.path.normpath(q_table_file)

            with open(q_table_file, 'rb') as f:
                q_dict = pickle.load(f)
            self.q_table = defaultdict(lambda: np.zeros(len(self.actions)), q_dict)
            print(f"Q-table cargada desde: {q_table_file}")
            print(f"Estados cargados: {len(self.q_table)}")

        except FileNotFoundError:
            print("No se encontró la Q-table entrenada. Se inicia una tabla vacía.")
            self.q_table = defaultdict(lambda: np.zeros(len(self.actions)))

        # (debug)
        print(f"[DEBUG] Archivo de Q-table usado: {q_table_file}")

    
    def discretize_state(self, state):
        """
        Discretiza el estado continuo en una tupla (player_y, vel, dist, pipe_top).
        """
        # Extraemos variables del entorno
        player_y = state['player_y']                      # Altura del pájaro (0–512 aprox)
        player_vel = state['player_vel']                  # Velocidad vertical (-10 a +10)
        next_pipe_dist = state['next_pipe_dist_to_player']# Distancia horizontal al tubo (0–300)
        next_pipe_top = state['next_pipe_top_y']          # Y del borde superior del tubo (100–400)

        # --- Discretización más fina ---
        player_y_bin = int(player_y // 25)                # antes 50 (≈10) → ahora más sensible (≈20 bins)
        player_vel_bin = int((player_vel + 10) // 2)      # antes 4 (≈5) → mejor resolución (≈10 bins)
        pipe_dist_bin = int(next_pipe_dist // 20)         # antes 30 (≈10) → mejor (≈15 bins)
        pipe_top_bin = int(next_pipe_top // 20)           # antes 40 (≈10) → mejor (≈20 bins)
        
        # Limitar los valores al máximo bin posible
        player_y_bin = min(player_y_bin, 20)
        player_vel_bin = min(max(player_vel_bin, 0), 10)
        pipe_dist_bin = min(pipe_dist_bin, 15)
        pipe_top_bin = min(pipe_top_bin, 20)

        # Bloque para ver cómo discretiza (solo debug)
        if np.random.rand() < 0.0005:  # imprime solo en 0.05% de los pasos
            print(f"[DEBUG] Estado continuo → y={player_y:.1f}, vel={player_vel}, dist={next_pipe_dist:.1f}, top={next_pipe_top:.1f}")
            print(f"[DEBUG] Estado discreto  → ({player_y_bin}, {player_vel_bin}, {pipe_dist_bin}, {pipe_top_bin})")

        # Devolver una tupla inmutable (clave para la Q-table)
        return (player_y_bin, player_vel_bin, pipe_dist_bin, pipe_top_bin)


    def act(self, state):
        """
        Selecciona una acción usando política epsilon-greedy.
        """
        discrete_state = self.discretize_state(state)

        # Exploración vs. explotación
        if np.random.rand() < self.epsilon:
            # acción aleatoria
            action = np.random.choice(self.actions)
        else:
            # acción con mayor Q-valor
            q_values = self.q_table[discrete_state]
            action = self.actions[np.argmax(q_values)]

        return action


    def update(self, state, action, reward, next_state, done):
        """
        Actualiza la Q-table usando la regla de Q-learning.
        """
        discrete_state = self.discretize_state(state)
        discrete_next_state = self.discretize_state(next_state)
        action_idx = self.actions.index(action)
        # Inicializar si el estado no está en la Q-table
        if discrete_state not in self.q_table:
            self.q_table[discrete_state] = np.zeros(len(self.actions))
        if discrete_next_state not in self.q_table:
            self.q_table[discrete_next_state] = np.zeros(len(self.actions))
        current_q = self.q_table[discrete_state][action_idx]
        max_future_q = 0
        if not done:
            max_future_q = np.max(self.q_table[discrete_next_state])
        new_q = current_q + self.lr * (reward + self.gamma * max_future_q - current_q)
        self.q_table[discrete_state][action_idx] = new_q

    def decay_epsilon(self):
        """
        Disminuye epsilon para reducir la exploración con el tiempo.
        """
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def save_q_table(self, path):
        """
        Guarda la Q-table en un archivo usando pickle.
        """
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(dict(self.q_table), f)
        print(f"Q-table guardada en {path}")

    def load_q_table(self, path):
        """
        Carga la Q-table desde un archivo usando pickle.
        """
        import pickle
        try:
            with open(path, 'rb') as f:
                q_dict = pickle.load(f)
            self.q_table = defaultdict(lambda: np.zeros(len(self.actions)), q_dict)
            print(f"Q-table cargada desde {path}")
        except FileNotFoundError:
            print(f"Archivo Q-table no encontrado en {path}. Se inicia una nueva Q-table vacía.")
            self.q_table = defaultdict(lambda: np.zeros(len(self.actions)))
