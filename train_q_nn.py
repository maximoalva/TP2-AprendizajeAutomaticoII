import numpy as np
import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# --- Cargar Q-table entrenada ---
QTABLE_PATH = "C:/Users/arand/OneDrive/Documentos/IA/AA2/TP2/tp2-ej2-template/flappy_birds_q_table_final.pkl"
with open(QTABLE_PATH, 'rb') as f:
    q_table = pickle.load(f)

# --- Preparar datos para entrenamiento ---
# Convertir la Q-table en X (estados) e y (valores Q para cada acción)
X = []  # Estados discretos
y = []  # Q-values para cada acción
for state, q_values in q_table.items():
    if q_values[0] == q_values[1] == 0:
        continue
    X.append(state)         # estado = tupla 4 int
    y.append(q_values)      # 2 valores Q
X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.float32)

print("Dimensión X:", X.shape)  # ej: (20000, 4)
print("Dimensión y:", y.shape)  # ej: (20000, 2)


model = keras.Sequential([
    layers.Input(shape=(4,)),
    layers.Dense(128, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(2, activation='linear')
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0005),
    loss='mse'
)

history = model.fit(
    X, y,
    batch_size=64,
    epochs=250,
    validation_split=0.1,
    verbose=1
)

# --- Mostrar métricas ---
print("Loss final:", history.history["loss"][-1])
print("Val Loss:", history.history["val_loss"][-1])

# --- Guardar el modelo entrenado ---
model.save('flappy_q_nn_model.keras')
print('Modelo guardado como TensorFlow SavedModel en flappy_q_nn_model/')

# --- Notas para los alumnos ---
# - Puedes modificar la arquitectura de la red y los hiperparámetros.
# - Puedes usar la red entrenada para aproximar la Q-table y luego usarla en un agente tipo DQN.
# - Si tu estado es una tupla de enteros, no hace falta normalizar, pero puedes probarlo. --> NO MEJORÓ
# - Si tienes dudas sobre cómo usar el modelo para predecir acciones, consulta la documentación de Keras.
