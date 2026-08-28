from agentes.base import Agent
import numpy as np
import tensorflow as tf
from tensorflow import keras

class NNAgent(Agent):
    """
    Agente que utiliza una red neuronal entrenada para aproximar la Q-table.
    La red debe estar guardada como TensorFlow SavedModel.
    """
    def __init__(self, actions, game=None, model_path='flappy_q_nn_model.keras'):
        super().__init__(actions, game)

        # Cargar el modelo Keras
        self.model = keras.models.load_model(model_path)

    # --- Misma discretización que QAgent ---
    def discretize_state(self, state):

        player_y = state['player_y']
        player_vel = state['player_vel']
        next_pipe_dist = state['next_pipe_dist_to_player']
        next_pipe_top = state['next_pipe_top_y']

        player_y_bin = int(player_y // 25)
        player_vel_bin = int((player_vel + 10) // 2)
        pipe_dist_bin = int(next_pipe_dist // 20)
        pipe_top_bin = int(next_pipe_top // 20)

        player_y_bin = min(player_y_bin, 20)
        player_vel_bin = min(max(player_vel_bin, 0), 10)
        pipe_dist_bin = min(pipe_dist_bin, 15)
        pipe_top_bin = min(pipe_top_bin, 20)

        return (player_y_bin, player_vel_bin, pipe_dist_bin, pipe_top_bin)
    
        # --- Acción basada en la red ---
    def act(self, state):

        s = np.array(self.discretize_state(state), dtype=np.float32).reshape(1, -1)

        q_values = self.model.predict(s, verbose=0)[0]

        action_idx = int(np.argmax(q_values))

        return self.actions[action_idx]

