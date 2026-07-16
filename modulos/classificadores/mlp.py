import time
from sklearn.neural_network import MLPClassifier
import numpy as np

# Treinando com MLP
def treinar_mlp(caracteristicas, rotulos):
    print('Treinando o modelo MLP...')
    modelo_mlp = MLPClassifier(
        random_state=1,
        # Quantidade de camadas ocultas e numero de neurônios em cada
        hidden_layer_sizes=(5000),
        max_iter=1000
    )
    startTime = time.time()
    modelo_mlp.fit(caracteristicas, rotulos)
    elapsedTime = round(time.time() - startTime, 2)
    print(f'Treinamento realizado em {elapsedTime}s')
    return modelo_mlp

# Testando com MLP
def testar_mlp(modelo_mlp, caracteristicas):
    print('Iniciando previsão...')
    startTime = time.time()
    rotulos_previstos = modelo_mlp.predict(caracteristicas)
    elapsedTime = round(time.time() - startTime,2)
    print(f'Previsão encerrada em {elapsedTime}s')
    return rotulos_previstos

# Ajusntando amostras zero
def ajustar_amostras_zero(X):
    '''
    Ajusta amostras que contêm todos os valores zero para garantir que pelo menos uma categoria estejja representada

    Args:
        X (numpy array): Array numpy contendo as amostras a serem ajustadas.

    Return:
        X_ajustado (numpy array): Array numpy com as amostras ajustadas;
    '''
    # Identifica amostras com todos os valores zero
    amostras_zero = np.all(X == 0, axis=1)

    # Ajusta amoostras com todos os valores zero
    X_ajustado = np.copy(X)
    X_ajustado[amostras_zero, 0] = 1 # Define o primeiro bit como 1 para amostras com todos os valores zeros

    return X_ajustado