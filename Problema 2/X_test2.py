from ple.games.flappybird import FlappyBird
from ple import PLE
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from agentes.dq_agent import QAgent



# --- Configuración del entorno ---
game = FlappyBird()
env = PLE(game, display_screen=True, fps=30)
env.init()

actions = env.getActionSet()
print(f"Acciones posibles: {actions}")

# --- Inicializar el agente ---
agent = QAgent(actions, game=game, epsilon=0.0)  # epsilon=0 para comportamiento determinista

# --- Loop de testeo ---
num_episodes = 3  # podés aumentar si querés observar más
for ep in range(num_episodes):
    env.reset_game()
    total_reward = 0
    done = False
    step = 0
    state = env.getGameState()

    print(f"\n===== Episodio {ep + 1} =====")
    print(f"Estado inicial: {state}")

    while not done:
        step += 1
        # Acción del agente
        action = agent.act(state)

        # Ejecutar acción en el entorno
        reward = env.act(action)
        next_state = env.getGameState()
        done = env.game_over()

        total_reward += reward

        # Muestra información de control
        discrete_state = agent.discretize_state(state)
        print(f"Paso {step}: estado_discreto={discrete_state}, acción={action}, recompensa={reward}")

        # Avanzar
        state = next_state
        time.sleep(0.03)

    print(f"Recompensa total del episodio: {total_reward}")
