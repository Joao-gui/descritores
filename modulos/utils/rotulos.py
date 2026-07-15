# Este módulo possui funções utilitárias para
# codificação e decodiuficação de rótulos

import numpy as np
from sklearn import preprocessing

# Codificar com LabelEncoder
def codificar_rotulos_label(rotulos):
    '''
    Codifica os rótulos de caregorias usando LabelEncoder

    Args:
        rotulos (list de str): Kusta de rótulos de categorias

    Return:
        rotulos_codificados (numpy ndarray): Array de rótulos codificados como inteiros.
        encoder (LabelEncoder): O objjeto LabelEncoder usado para a codificação.
    '''
    # Cria uma instância do LabelEncoder
    encoder = preprocessing.LabelEncoder()

    # Ajusta o encoder aos rótulos e os transforma em inteiros
    rotulos_codificados = encoder.fit_transform(rotulos)
    print('Rótulos codificados com LabelEncoder')
    return rotulos_codificados, encoder

# Codificar com OneHotEncoder
def codificar_rotulos_onehot(rotulos):
    '''
    Codifica os rótulos de categorias usando OneeHotEncoder

    Args:
        rotulos (list de str): Lista de rótulos de categorias.

    Return:
        rotulos_onehot (numpy ndarray): Array 2D de rótulos codificados como vetores OneHot.
        encoder (OneHotEncoder): O objeto OneHotEncoder usado para a codificação.
    '''
    # Converte os rótulos para um array 2D (requisito do OneHotEncoder)
    rotulos = np.array(rotulos).reshape(-1,1)

    # Cria uma instância do OneHotEncoder
    encoder = preprocessing.OneHotEncoder(sparse_output=False) # sparse_output=False retorna um array denso

    # Ajusta o encoder aos rótulos e os transforma em OneHot
    rotulos_codificados = encoder.fit_transform(rotulos)
    print('Rótulos codificados com OneHotEncoder')
    return rotulos_codificados, encoder