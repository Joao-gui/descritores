import time
from sklearn import svm

# Treinando com SVM
def treinar_svm(caracteristicas, rotulos):
    print('Treinando o modelo SVM...')
    modelo_svm = svm.SVC(kernel='linear', C=1, random_state=42)
    startTime = time.time()
    modelo_svm.fit(caracteristicas, rotulos)
    elapsedTime = round(time.time() - startTime, 2)
    print(f'Treinamento encerrado em {elapsedTime}s')
    return modelo_svm

# Testando com SVM
def testar_svm(modelo_svm, caracteristicas):
    print('Iniciando previsão...')
    startTime = time.time()
    rotulos_previstos = modelo_svm.predict(caracteristicas)
    elapsedTime = round(time.time() - startTime,2)
    print(f'Previsão encerrada em {elapsedTime}s')
    return rotulos_previstos