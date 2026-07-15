import os
#import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics

# Verifica e cria diretorios
def verifica_e_cria_diretorios(caminho_arquivo):
    '''
    Verifica se os diretórios no caminho fornecido existem e os cria se necessário.

    Args:
        caminho_arquivo (str): Caminho completo do arquivo, incluindo o nome do arquivo
    '''
    diretorio = os.path.dirname(caminho_arquivo)
    if not os.path.exists(diretorio):
        os.makedirs(diretorio)
        print(f'Diretório criado: {diretorio}')
    else:
        print(f'Diretório já existe: {diretorio}')

# Matriz Confusão
def matriz_confusao(nomes_das_classes, rotulos_verdadeiros, rotulos_previstos, caminho_arquivo):
    # Calcula a matriz de confusão
    conf_matriz = metrics.confusion_matrix(rotulos_verdadeiros, rotulos_previstos)
    # Calcula a acurácia do modelo em percentual
    acuracia = metrics.accuracy_score(rotulos_verdadeiros, rotulos_previstos)*100
    # Define o tamanho da fonte para a matriz de confusão e rótulos
    sns.set_theme(font_scale=1.2) # Ajuste o valor conforme necessário para aumentar a fonte
    # Gera a figura da matriz de confusão usando o pacote seaborn
    plt.figure(figsize=(4,3))
    sns.heatmap(conf_matriz, annot=True, cmap='Blues', fmt='g', cbar=False,
                annot_kws={'size': 12}, linewidths=0.4, square=True)
    # Adiciona os nomes das classes, rótulos, título, subtítulo
    plt.xticks(np.arange(len(nomes_das_classes)) + 0.5, nomes_das_classes, rotation=0, ha='center', fontsize=14)
    plt.yticks(np.arange(len(nomes_das_classes)) + 0.5, nomes_das_classes, rotation=0, va='center', fontsize=14)
    plt.xlabel('Rótulos Previstos', fontsize=14)
    plt.ylabel('Rótulos Verdadeiros', fontsize=14)
    plt.title('Matriz de Confusão', fontsize=18, weight='bold', x=0.25, y=1.15)
    plt.suptitle(f'Acurácia do Modelo: {acuracia:.2f}%', fontsize=14, x=0.37, y=0.98)
    verifica_e_cria_diretorios(caminho_arquivo)
    # Salva a figura da matriz de confusão
    plt.savefig(caminho_arquivo, dpi=300)
    plt.show()

# Relatorio classificação
def relatorio_classificacao(nomes_das_classes, rotulos_verdadeiros, rotulos_previstos, caminho_arquivo):
    # Gera o relatório de classificação
    report = metrics.classification_report(rotulos_verdadeiros, rotulos_previstos, target_names=nomes_das_classes, output_dict=True)

    # Converte o relatório para um DataFrame
    report_df = pd.DataFrame(report).transpose()

    # Define o tamanho da figura e o estilo do seaborn
    plt.figure(figsize=(6, len(report_df) / 2))
    sns.set_theme(style='whitegrid')

    # Cria a tabela utilizando seaborn
    sns.heatmap(report_df, annot=True, cmap='Blues', fmt='.2f', cbar=False, annot_kws={'size': 12})

    # Adiciona uma linha em branco antes do título
    plt.subplots_adjust(top=0.85)
    # Ajusta os rótulos e o título
    plt.title('Relatório de Classificação', fontsize=16, weight='bold', x=0.5, y=1.05)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12, rotation=0)
    verifica_e_cria_diretorios(caminho_arquivo)
    # Salva a figura da tabela do relatório de classificação
    plt.savefig(caminho_arquivo, dpi=300)
    plt.show()