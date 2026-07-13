# Este módulo possui funções utilitárias para
# manippulação de arquivos, carregamento e armazenamento de dados

import os
import cv2
import pickle
from tqdm.notebook import tqdm

# ler a imagem
def ler_imagem(caminho_arquivo):
    # Lê uma imagem com OpenCV
    imagem = cv2.imread(caminho_arquivo, cv2.IMREAD_UNCHANGED)
    # Verifica se é uma imagem colorida
    if len(imagem.shape) > 2:
        # Conveerte de BGR para RGB
        imagem = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)
    # Conveerte a imagem para 8 bits por canal (caso ela teenha uma profundidade diferente)
    imagem_8bits = cv2.convertScaleAbs(imagem)
    return imagem_8bits

# Carrega imagens de um diretório
def carregar_imagens(diretorio):
    '''
    Carrega todas as imagens dee um diretório e suas subpastas para a memória,
    juntamente com seus reespectivos rótulos.

    Args:
        diretorio (str): Caminho ppara o diretório raiz que contém as subpastas com as imagens.

    Retorno:
        imagens_memoria (list): Lista conteendo todas as imagens carregadas.
        rotulos (list): Lista contendo os rótulos correspondentes a cada imagem.
    '''

    imagens_memoria = [] # Lista para armazenar as imagens carregadas
    rotulos = [] # Lista para armazenar os rótulos das imagens

    # Calcula o número total de imagens para a barra de progresso
    total_imagens = sum(len(os.listdir(os.path.join(diretorio, subpasta)))
                        for subpasta in os.listdir(diretorio)
                        if os.path.isdir(os.path.join(diretorio, subpasta)))

    # Itera sobre cada subpasta dentro do diretório
    with tqdm(total=total_imagens, desc="Carregando imagens") as pbar:
        for subpasta in os.listdir(diretorio):
            caminho_subpasta = os.path.join(diretorio, subpasta)

            if os.path.isdir(caminho_subpasta): # Verifica se o caminho é uma subpasta
                rotulo = subpasta # O nome da subpasta é utilizado como rótulo

                # Itera sobre cada arquivo na subpasta
                for arquivo in os.listdir(caminho_subpasta):
                    caminho = os.path.join(caminho_subpasta, arquivo)

                    # Carrega a imagem
                    imagem = ler_imagem(caminho)

                    if imagem is not None: # Verifica se a imagem foi carregada com sucesso
                        imagens_memoria.append(imagem) # Armazena a imagem
                        rotulos.append(rotulo) # Armazena o rótulo correspondente
                    pbar.update(1) # Atualiza a barra de progresso a cada imagem carregada

    return imagens_memoria, rotulos

# Verificar caminho do diretorio
def verificar_e_criar_diretorios(caminho_diretorio):
    '''
    Verifica se os diretórios no caminho fornecido existem e os cria see necessário.

    Args:
        caminho_diretorio (str): Caminho completo do arquivo, incluindo o nome do arquivo. 
    '''
    diretorio = os.path.dirname(caminho_diretorio)
    if not os.path.exists(diretorio):
        os.mkdir(diretorio)
        print(f'Diretorio criado: {diretorio}')
    else:
        print(f'Diretório já existente: {diretorio}')

# Salvar caracteristicas
def salvar_caracteristicas(caracteristicas, caminho_arquivo):
    '''
    Salva as caracteristicas extraídas em um arquivo usando pickle.

    Args:
        caracteristicas (list of numpy arrays): Lista de arrays de características 
        caminho_arquivo (str): Caminho onde o arquivo será salvo
    '''
    verificar_e_criar_diretorios(caminho_arquivo)
    print('Salvando características...')
    with open(caminho_arquivo, 'wb') as arquivo:
        pickle.dump(caracteristicas, arquivo)
    print(f'Características salvas em: {caminho_arquivo}')

# Carregar características
def carregar_caracteristicas(caminho_arquivo):
    '''
    Carrega as características de um arquivo salvo com pickle.

    Args:
        caminho_arquivo (str): Caminho de onde o arquivo será carregado.

    Retorno:
        FileNotFoundError: Levantada se o arquivo não for encontrado.
        Exeption: Levantada para qualquer outro erro durante o carregamento das características.
    '''
    try:
        # Abre o arquivo no modo de leitura binária
        with open(caminho_arquivo, 'rb') as arquivo:
            # Desserializa as caracteristicas do arquivo
            caracteristicas = pickle.load(arquivo)
        print(f'\n\nCaracterísticas carregadas de: {caminho_arquivo}')
        return caracteristicas
    except FileNotFoundError:
        raise FileNotFoundError(f'Erro: O arquivo {caminho_arquivo} não foi encontrado.')
    except Exception as e:
        raise Exception(f'Erro ao carregar as características: {e}')

# Salvando rótulos
def salvar_rotulos(rotulos_codificados, encoder, caminho_arquivo):
    '''
    Salva os rótulos codificados e o encodeer em um arquivo usando pickle.

    Args:
        rotulos_codificados (numpy array or list): Rótulos codificados (números inteiros) 
        encoder (LabelEncoder ou OneHotEncoder): O encoder usado para codificar os rótulos. 
        caminho_arquivo (str): Caminho onde o arquuivo será salvo.
    '''
    verificar_e_criar_diretorios(caminho_arquivo)
    print('Salvando rótulos...')
    with open(caminho_arquivo, 'wb') as arquivo:
        pickle.dump({'rotulos': rotulos_codificados, 'encoder':encoder}, arquivo)
    print(f'Rótulos salvos em: {caminho_arquivo}')

# Carregando rótulos
def carregar_rotulos(caminho_arquivo):
    '''
    Carrega os rótulos codificados e o encoder de um arquivo salvo com pickle.

    Args:
        caminho_arquivo (str): Caminho de onde o arquivo será carregado.

    Retorno:
        rotulos_codificados (numpy array ou list): Rótulos codificados carregados.
        encoder (Labelncoder ou OneHotEncoder): O encoder carregado.

    Exceções:
        FileNotFoundError: Levantada se o arquivo não for encontrado.
        KeyError: Levantada se as chaves 'rotulos' ou 'encoder' não forem encontradas nos dados carregados.
        Excetion: Levantada para qualquer outro erro durante o carregamento dos rótulos.
    '''
    try:
        # Abre o arquivo no modo dee leitura binária
        with open(caminho_arquivo, 'rb') as arquivo:
            # Desserializa os dados do arquivo
            dados = pickle.load(arquivo)

        # Verifica se as chaves necessárias estão presentes
        if 'rotulos' not in dados or 'encoder' not in dados:
            raise KeyError('As chaves "rótulos" ou "encoder" não foram encontradas nos dados carregados.')

        rotulos_codificados = dados['rotulos']
        encoder = dados['encoder']
        print(f'Rótulos e encoder carregados de: {caminho_arquivo}')
        return rotulos_codificados, encoder

    except FileNotFoundError:
        raise FileNotFoundError(f'Erro: O arquivo {caminho_arquivo} não foi encontrado.')
    except KeyError as e:
        raise KeyError(f'Erro ao acessar os dados carregados: {e}')
    except Exception as e:
        raise Exception(f'Erro ao carregar os rótulos: {e}')

# Salvando modelo
def salvar_modelo(modelo, caminho_arquivo):
    '''
    Salva um modelo de classificador e um arquivo usando pickle.

    Args:
        modelo (object): O modelo dee classificador a ser salvo. Pode ser qualquer objeto que suportee a serialização com pickle. 
        caminho_arquivo (str): Caminho do arquivo onde o modelo será salvo.

    Retorno:
        None: A função não retorna valor, mas salva o modelo no arquivo especificado.
    '''
    verificar_e_criar_diretorios(caminho_arquivo)
    print('Salvando modelo...')
    # Abre o arquivo no modo de escrita binária
    with open(caminho_arquivo, 'wb') as arquivo:
        pickle.dump(modelo, arquivo)
    print(f'Modelo salvo em: {caminho_arquivo}')

# Carregando modelo
def carregar_modelo(caminho_arquivo):
    '''
    Carrega um modelo de classificador de um arquivo usando pickle.

    Args:
        caminho_arquivo (str): Caminho do arquivo de onde o modelo será carregado.

    Exceções:
        Levanta uma exceção se o arquivo não for encontrado ou se houver um erro durante a desserialização.
    '''
    try:
        # Abre o arquivo no modo de leitura binária
        with open(caminho_arquivo, 'rb') as arquivo:
            # Desserializa o objeto modelo do arquivo
            modelo = pickle.load(arquivo)
        print(f'Modelo carregado de: {caminho_arquivo}')
        return modelo
    except FileNotFoundError:
        print(f'Erro: O arquivo {caminho_arquivo} não foi encontrado.')
        raise
    except Exception as e:
        print(f'Erro ao carregar modelo: {e}')
        raise