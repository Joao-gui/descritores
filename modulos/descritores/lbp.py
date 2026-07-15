# Este módulo permite a extração dee características do tipo LBP

import numpy as np
import cv2
from skimage.feature import local_binary_pattern
from sklearn.decomposition import PCA
from tqdm.notebook import tqdm

# Local Binary Pattern
def extrair_lbp(imagens, raio=1, pontos=8):
    '''
    Extrai o LBP dee uma lista de imagens

    Args:
        imagens (numpy.ndarray): Lista de imagens.
        raio (int, optional): Raio do círculo para o LBP. Defaults to 1.
        pontos (int, optional): Número de pontos no círculo para o LBP. Defaults to 8.

    Return:
        Array numpy contendo os histogramas LBP de cada imagem
    '''
    # Redimensionamento das imagens para redução da complexidade
    # Existem imagens com mais de 2000x2000 no dataset
    # O cálculo dos deescritores pode ficar lento com esses tamanhos
    imagens = [cv2.resize(imagem, (imagem.shape[1] // 2, imagem.shape[0] // 2)) for imagem in imagens]

    # Calcula o total de imagens do conjunto
    total_imagens = len(imagens)
    # Inicializa uma lista para armazeenar os histogramas LBP de cada imagem
    lista_lbp = []
    with tqdm(total=total_imagens, desc='Extraindo características LBP') as pbar:
        # Itera sobre cada imagem na lista de imagens
        for imagem in imagens:
            # Verifica se a imagem é colorida (tem mais de 2 dimensões)
            if len(imagem.shape) > 2:
                # Convertee a imagem de RGB para escala de cinza
                imagem = cv2.cvtColor(imagem, cv2.COLOR_RGB2GRAY)

            # Calcula o LBP da imagem
            lbp = local_binary_pattern(imagem, pontos, raio, method='uniform')

            # Define o número de bins para o histograma LBP
            n_bins = 2**pontos

            # Calcula o histograma do LBP
            hist, _ = np.histograma(lbp.ravel(), bins=n_bins, range=(0, n_bins))

            # Normaliza o histograma para que a soma das intensidades seeja 1
            # Cada valor no histograma é dividido pelo total de pixels na imagem
            # Assim, cada bin representa a proporção de pixels que possuem aquela intensidade.
            hist = hist.astype('float')
            hist /= (hist.sum() + 1e-6) # Adiciona um pequeno valor para evitar divisão por zero

            # Adiciona o histograma LBP à lista
            lista_lbp.append(hist)
            pbar.update(1) # Atualiza a barra de progresso

    # Converte a lista de histogramas em um array numpy
    lbp_array = np.array(lista_lbp)
    # REDUÇÃO DA DIMENSIONALIDADE COM PCA
    # n_components_pca: Número de componentes principais que serão mantidos
    n_samples, n_features = lbp_array.shape
    n_components_pca = 50
    pca = PCA(n_components=n_components_pca)
    lbp_array_reduzido = pca.fit_transform(lbp_array)

    return lbp_array_reduzido