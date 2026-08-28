# a)	Descripción de la ingeniería de características sobre el estado del juego
El entorno de Flappy Bird es continuo: la posición del jugador, la velocidad vertical y la posición relativa al siguiente tubo pueden tomar miles de valores posibles. 
Con la discretización buscamos determinar un espacio finito para transformar el estado continuo en un estado discreto manejable y así poder construir la Qtable.
Desarrollamos la función “discretize_state”, donde cada variable continua del estado fue discretizada mediante bins uniformes, definiendo intervalos que representan rangos específicos.
Con esto, el estado final queda representado como un vector de índices discretos, permitiendo mapear cada combinación a una entrada de la matriz Q.
Al inicio tuvimos algunos inconvenientes con el tamaño de los bins, que hacían que el agente no aprendiera bien, es decir, teníamos los estados mal discretizados. Muchos estados distintos del entorno caían en la misma celda de la Q-table, por lo que el agente no podía distinguir bien cuándo tenía que saltar y cuándo no. Esto hizo que la política se volviera demasiado general y repetitiva, por ejemplo: “siempre saltaba”.
Lo que hicimos luego fue aumentar la cantidad de bines haciendo una discretización más fina y con esto, logramos un mejor entrenamiento.
Limitamos también los valores de los bins para que los estados siempre caigan dentro del espacio de estados entrenado, evitar valores extremos que generen nuevos estados no vistos y controlar el tamaño de la Q-table.
# b)	Análisis y comparación de los resultados obtenidos para los diferentes agentes.
En el trabajo implementamos dos agentes para el entorno dado:
1.	Un agente de Q-Learning con discretización del espacio de estados.
2.	Un agente de Redes Neuronales a partir de la Q-table.
A continuación presentamos los resultados y un breve análisis de como llegamos a ellos.

## 1. Resultados del agente Q-Learning (30 episodios)
El agente Q-Learning fue entrenado utilizando un espacio de estados discretizado manualmente (4 variables, cada una dividida en cierto número de bins). Sus resultados fueron:
- Score promedio: 95.63
- Score máximo: 278
- Supervivencia promedio: 3645 steps
- Recompensa promedio: 90.6
Puntajes alcanzados durante el entrenamiento: bajo a moderado. Este desempeño representa un agente que aprendió una política razonablemente buena dentro del espacio limitado de estados disponibles. Cuenta con limitaciones en escenarios más dinámicos debido a la pérdida de información causada por la discretización.

## 2. Resultados del agente con Red Neuronal – Versión PREVIA (30 episodios)
En la primera versión que construimos del agente basado en redes neuronales entrenábamos tomando todas las entradas de la Q-table, incluyendo las filas donde ambos valores Q eran cero. Eso significa que:
- La mayor parte de la tabla eran estados nunca visitados (Q=[0,0]).
- La red se entrenaba sobre una enorme cantidad de datos sin información, que actuaban como ruido.
- Esto hacía que el modelo aprendiera un promedio que se acercaba a “todo vale cero”.
Los resultados fueron muy bajos:
- Score promedio: 14.13
- Score máximo: 41
- Supervivencia promedio: 571 steps
- Recompensa promedio: 9.1
Era evidente que este agente no lograba aproximar correctamente la política aprendida por Q-Learning.
A partir de lo discutido en clase descubrimos que el modelo estaba aprendiendo a partir de demasiados datos nulos, lo cual generaba:
-	Sesgo masivo hacia Q ≈ 0: La red veía miles de estados con Q=[0,0].
-	Promedios desplazados: Como en la mayoría de los datos Q=0, la red aproximaba valores muy bajos incluso para estados importantes.
Por esto se introdujo como cambio eliminar del dataset todos los estados jamás visitados por Q-Learning, de modo que la red solo aprenda sobre el conocimiento real adquirido por el agente.

## 3. Resultados del agente con Red Neuronal – Versión AJUSTADA (10 episodios)
Tras entrenar el modelo con estas modificaciones el desempeño del agente mejoró de forma abismal.
Se corrieron solo 10 episodios por limitaciones de tiempo (ya que este agente tarda más en ejecutarse), pero aun así los resultados muestran la magnitud del cambio:
- Score promedio: 680.3
- Score máximo: 1423
- Supervivencia promedio: 25684 steps
- Recompensa promedio: 675.3
Esto representa un incremento de rendimiento de más de 4800% respecto a la versión previa del modelo neuronal.
Incluso con pocos episodios, el comportamiento observado refleja una política mucho más estable y eficiente que la del Q-Learning clásico.

# Conclusión final
- El Q-Learning funciona bien dentro de un espacio discreto reducido, pero tiene un límite natural de precisión.
- El primer modelo de Red Neuronal falló debido a que se entrenó con una enorme cantidad de datos sin información (Q=[0,0]), lo que sesgó el aprendizaje.
- Luego del ajuste (eliminando estados irrelevantes del dataset) el modelo neuronal finalmente pudo aprender una aproximación válida de la política, superando ampliamente al modelo tabular.
Con los datos corregidos, el enfoque con redes neuronales no solo iguala, sino que supera significativamente al agente de Q-Learning.
