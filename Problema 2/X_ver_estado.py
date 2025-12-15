from ple import PLE
from ple.games.flappybird import FlappyBird

# Crear el juego
game = FlappyBird()

# Crear el entorno PLE
env = PLE(game, fps=30, display_screen=False)

# Inicializar
env.init()

# Obtener el conjunto de acciones disponibles
actions = env.getActionSet()
print("Acciones posibles:", actions)

# Reiniciar el juego
env.reset_game()

# Obtener el estado inicial
state = env.getGameState()
print("Estado inicial del juego:")
print(state)
