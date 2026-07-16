import time
from sklearn.ensemble import RandomForestClassifier

# Treinando com Random Foorest
def treinar_rf(caracteristicas, rotulos):
    print('Treinando o modelo Random Forest...')
    modelo_rf = RandomForestClassifier(n_estimators=100, random_state=42)
    startTime = time.time()
    modelo_rf.fit(caracteristicas, rotulos)
    elapsedTime = round(time.time() - startTime,2)
    print(f'Treinamento encerrado em {elapsedTime}s')
    return modelo_rf

# Testando com Random Forest
def testar_rf(modelo_rf, caracteristicas):
    print('Iniciando previsão...')
    startTime = time.time()
    rotulos_previstos = modelo_rf.predict(caracteristicas)
    elapsedTime = round(time.time() - startTime,2)
    print(f'Previsão encerrada em {elapsedTime}s')
    return rotulos_previstos