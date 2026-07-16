import time
from sklearn.neighbors import KNeighborsClassifier

# Treinando com KNN
def treinar_knn(caracteristicas, rotulos, n_neighbors=5):
    print('Treinando o modelo KNN...')
    modelo_knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    startTime = time.time()
    modelo_knn.fit(caracteristicas, rotulos)
    elapsedTime = round(time.time() - startTime, 2)
    print(f'Treinamento realizado em {elapsedTime}s')
    return modelo_knn

# Testando com KNN
def testar_knn(modelo_knn, caracteristicas):
    print('Iniciando previsão...')
    startTime = time.time()
    rotulos_previstos = modelo_knn.predict(caracteristicas)
    elapsedTime = round(time.time() - startTime, 2)
    print(f'Previsão encerrada em {elapsedTime}s')
    return rotulos_previstos