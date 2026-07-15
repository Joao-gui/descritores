# Este módulo permite a extração de características do tipo histograma

import cv2
import numpy as np
from sklearn.decomposition import PCA
from tqdm.notebook import tqdm

# Extrai histograma escala de cinza
def extrai_histograma_escala_cinza(imagens):
    '''
    Extrai o histograma em escala de cinza de uma lista de imagens.

    Args:
        imagens (numpy.ndarray): Lista de imagens em formato array numpy.

    Return:
        histogramas_reduzidos (numpy.ndarray): Array numpy contendo os histogramas de cada imagem.
    '''
    # Inicializa uma lista para armazenar os histogramas de cada imagem
    lista_histogramas = []
    # Calcula o total de imagens do conjunto
    total_imagens = len(imagens)
    with tqdm(total=total_imagens, desc='Extraindo características do tipo Histograma') as pbar:
        # Itera sobre cada imagem na lista de imagens
        for imagem in imagens:
            # Verifica se a imagem é colorida (tem mais de 2 dimensões)
            if len(imagem.shape) > 2:
                # Converte a imagem de RGB para escala de cinza
                imagem = cv2.cvtColor(imagem, cv2.COLOR_RGB2GRAY)

            # Define o número de bins para o histoograma (256 tons de cinza)
            bins = [256]

            # Calcula o histograma da imagem em escala de cinza
            hist = cv2.calcHist([imagem], [0], None, bins, [0, 256])

            # Normaliza o histograma para que a soma das intensidades seja 1
            # Cada valor no histograma é dividido pelo total de pixels na imagem
            # Assim, cada bin representa a proporção de pixels que possuem aquela intensidade
            cv2.normalize(hist, hist)

            # cv2.calcHist retorna um histograma 2D na forma (256, 1)
            # Por isso, achata-se o histograma em uma única dimensão (256, )
            lista_histogramas.append(hist.flatten())
            pbar.update(1) # Atualiza a barra de progresso
    # Converte a lista de histogramas em um array numpy
    histogramas_array = np.array(lista_histogramas)

    # REDUÇÃO DE DIMENSIONALIDADE COM PCA
    # n_components_pca: Número de componentes principais que serão mantidos
    n_samples, n_features = histogramas_array.shape
    n_components_pca = 50

    pca = PCA(n_components=n_components_pca)
    histogramas_reduzidos = pca.fit_transform(histogramas_array)

    return histogramas_reduzidos