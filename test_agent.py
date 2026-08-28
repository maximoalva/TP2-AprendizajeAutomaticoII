from ple.games.flappybird import FlappyBird
from ple import PLE
import time
import argparse
import importlib
import sys
import numpy as np


# --- Configuración del Entorno y Agente ---
# Inicializar el juego
game = FlappyBird()  # Usar FlappyBird en vez de Pong
env = PLE(game, display_screen=True, fps=30) # fps=30 es más normal, display_screen=True para ver


# Inicializar el entorno
env.init()

# Obtener acciones posibles
actions = env.getActionSet()

# --- Argumentos ---
parser = argparse.ArgumentParser(description="Test de agentes para FlappyBird (PLE)")
parser.add_argument('--agent', type=str, required=True, help='Ruta completa del agente, ej: agentes.random_agent.RandomAgent')
args = parser.parse_args()

# --- Carga dinámica del agente usando path completo ---
try:
    module_path, class_name = args.agent.rsplit('.', 1)
    agent_module = importlib.import_module(module_path)
    AgentClass = getattr(agent_module, class_name)
except (ValueError, ModuleNotFoundError, AttributeError):
    print(f"No se pudo encontrar la clase {args.agent}")
    sys.exit(1)

# Inicializar el agente
agent = AgentClass(actions, game)

# =====================================================
#              MÉTRICAS PARA LOS AGENTES
# =====================================================

EPISODES = 30 

scores = []
survival_times = []
total_rewards = []

print("\n==============================")
print(f"  Ejecutando {EPISODES} episodios")
print("==============================\n")

for ep in range(EPISODES):
    env.reset_game()
    agent.reset()
    state_dict = env.getGameState()
    done = False

    episode_reward = 0
    episode_score = 0
    steps = 0

    print(f"\n--- Episodio {ep + 1} ---")

    while not done:
        action = agent.act(state_dict)
        reward = env.act(action)
        state_dict = env.getGameState()
        done = env.game_over()

        episode_reward += reward
        steps += 1

        # Score = reward de +1
        if reward == 1.0:
            episode_score += 1

        time.sleep(0.01)

    print(f"Episodio terminado | Score: {episode_score} | Reward total: {episode_reward} | Steps: {steps}")

    scores.append(episode_score)
    total_rewards.append(episode_reward)
    survival_times.append(steps)

# =====================================================
#                RESULTADOS FINALES
# =====================================================

print("\n==============================")
print("      RESULTADOS FINALES      ")
print("==============================\n")

print(f"✔ Score promedio:          {np.mean(scores):.2f}")
print(f"✔ Score máximo:            {np.max(scores)}")
print(f"✔ Score mínimo:            {np.min(scores)}\n")

print(f"✔ Survival time promedio:  {np.mean(survival_times):.1f} steps")
print(f"✔ Survival time máximo:    {np.max(survival_times)}\n")

print(f"✔ Recompensa promedio:     {np.mean(total_rewards):.3f}\n")

print("Historial de scores:", scores)
