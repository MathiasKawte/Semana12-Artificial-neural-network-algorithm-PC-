import numpy as np
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([0, 0, 0, 1])

pesos = np.zeros(2)
bias = 0
learning_rate = 0.1
epocas = 20   

def activacion(x):
    return 1 if x >= 0 else 0

for epoch in range(epocas):
    for i in range(len(X)):
        salida = activacion(np.dot(X[i], pesos) + bias)
        error = y[i] - salida

        pesos += learning_rate * error * X[i]
        bias += learning_rate * error

print("===================================")
print("Perceptrón Simple")
print("===================================")
print("Pesos finales:", pesos)
print("Bias final:", bias)

print("\nClasificación:")

for i in range(len(X)):
    salida = activacion(np.dot(X[i], pesos) + bias)
    print(f"Entrada: {X[i]} -> Salida: {salida}")