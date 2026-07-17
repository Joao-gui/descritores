# 🖼️ Sistema de Classificação de Imagens

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg) ![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg) ![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg) ![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-orange.svg)

Sistema de visão computacional para classificação de imagens. O projeto combina diferentes descritores de imagem e algoritmos de machine learning para encontrar a melhor abordagem de classificação.

## 📋 Sumário

- Sobre o Projeto
- Estrutura do Projeto
- Tecnologias Utilizadas
- Funcionalidades
- Instalação
- Como Usar
- Pipeline do Sistema
- Resultados
- Licença

## 🎯 Sobre o Projeto

Este projeto foi desenvolvido para a disciplina de Visão Computacional e tem como objetivo classificar imagens em geral, porém para este projeto específico foi utilizado para imagens de raio-X, separado em duas categorias:

- COVID-19 (positivo para COVID-19)
- NORMAL (saudável)

O sistema permite comparar diferentes combinações de descritores de características e classificadores para encontrar o melhor modelo para o problema.

### Descritores de Características

| Descritor  | Descrição                                                     |
| ---------- | --------------------------------------------------------------- |
| Histograma | Distribuição de intensidade de pixels                         |
| HOG        | Histogram of Oriented Gradients - detecção de bordas e formas |
| LBP        | Local Binary Patterns - textura da imagem                       |
| SIFT       | Scale-Invariant Feature Transform - pontos-chave invariantes    |

### Classificadores

| Classificador | Descrição                      |
| ------------- | -------------------------------- |
| KNN           | K-Nearest Neighbors              |
| SVM           | Support Vector Machine           |
| MLP           | Multi-Layer Perceptron           |
| Random Forest | Floresta de árvores de decisão |

## 📁 Estrutura do Projeto

```
descritores/
├── datasets/                      # Conjuntos de dados
│   └── {seu_dataset}/
│       ├── train/
│       │   ├── {classe A}/             # Imagens classe A (treino)
│       │   └── {classe B}/             # Imagens classe B (treino)
│       └── test/
│           ├── {classe A}/             # Imagens classe A (teste)
│           └── {classe B}/             # Imagens classe B (teste)
│
├── features/                      # Características extraídas
│   └── caracteristicas/
│       ├── Histograma/
│       │   ├── treino.pkl
│       │   └── teste.pkl
│       ├── HOG/
│       │   ├── treino.pkl
│       │   └── teste.pkl
│       ├── LBP/
│       │   ├── treino.pkl
│       │   └── teste.pkl
│       └── SIFT/
│           ├── treino.pkl
│           └── teste.pkl
│
├── modelos/                       # Modelos treinados
│   ├── KNN/
│   │   ├── Histograma.pkl
│   │   ├── HOG.pkl
│   │   ├── LBP.pkl
│   │   └── SIFT.pkl
│   ├── SVM/
│   │   └── ...
│   ├── MLP/
│   │   └── ...
│   └── Random Forest/
│       └── ...
│
├── modulos/                       # Módulos do sistema
│   ├── descritores/               # Extração de características
│   │   ├── histograma.py
│   │   ├── hog.py
│   │   ├── lbp.py
│   │   └── sift.py
│   ├── classificadores/           # Algoritmos de ML
│   │   ├── knn.py
│   │   ├── svm.py
│   │   ├── mlp.py
│   │   └── random_forest.py
│   └── utils/                     # Funções auxiliares
│       ├── dados.py
│       ├── metricas.py
│       ├── rotulos.py
│       └── streamlit.py
│
├── resultados/                    # Resultados das avaliações
│   └── {seu_dataset}/
│       ├── Histograma/
│       │   ├── KNN/
│       │   │   ├── matriz_confusao.png
│       │   │   └── realtorio_calssificacao.png
│       │   └── ...
│       ├── HOG/
│       └── ...
│
├── main.py                        # Ponto de entrada
├── README.md
├── requirments.txt
├── .gitignore
└── LICENSE
```

## 🛠️ Tecnologias Utilizadas

| Tecnologia   | Versão | Finalidade                  |
| ------------ | ------- | --------------------------- |
| Python       | 3.8+    | Linguagem principal         |
| Streamlit    | 1.28+   | Interface web               |
| OpenCV       | 4.8+    | Processamento de imagens    |
| Scikit-learn | 1.3+    | Machine learning            |
| NumPy        | 1.24+   | Computação numérica      |
| Pandas       | 2.0+    | Manipulação de dados      |
| Matplotlib   | 3.7+    | Visualização de dados     |
| Seaborn      | 0.12+   | Visualização estatística |

## ✨ Funcionalidades

### 📂 Gerenciamento de Datasets

- Interface para selecionar diferentes datasets
- Visualização de informações do dataset (classes, quantidades)
- Suporte para estrutura `train/` e `test/`

### 🧬 Extração de Características

- 4 descritores diferentes disponíveis
- Barra de progresso durante a extração
- Salvamento automático em arquivos .pkl
- Redução de dimensionalidade com PCA
- Reutilização de características já extraídas

### 🤖 Treinamento de Modelos

- 4 classificadores disponíveis
- Medição do tempo de treinamento
- Salvamento organizado por classificador e descritor
- Interface intuitiva para seleção

### 📊 Avaliação e Resultados

- Matriz de confusão
- Relatório de classificação
- Métricas: Accuracy, Precision, Recall, F1-Score
- Salvamento automático de resultados
- Visualização interativa com Streamlit

## 🚀 Instalação

* Clone o repositório

```Shell
git clone https://github.com/Joao-gui/descritores.git
cd descritores
```

* Crie um ambiente virtua

```Shell
# Opção 1: Ambiente virtual (venv)
# Criar ambiente virtual
python -m venv venv

# Ativar ambiente (Linux/Mac)
source venv/bin/activate

# Ativar ambiente (Windows)
venv\Scripts\activate

# Opção 2: Ambiente Conda (Recomendado)
# Criar ambiente com Python 3.10
conda create -n {nome_ambiente} python=3.10

# Ativar ambiente
conda activate {nome_ambiente}

# (Opcional) Para desativar o ambiente
# conda deactivate
```

* Instale as dependências

```Shell
pip install -r requirements.txt
```

## 📖 Como Usar

1. Estrutura do Dataset

Organize seu dataset da seguinte forma:

```
datasets/
└── SeuDataset/
    ├── train/
    │   ├── classe1/
    │   │   ├── imagem1.jpg
    │   │   └── imagem2.jpg
    │   └── classe2/
    │       ├── imagem3.jpg
    │       └── imagem4.jpg
    └── test/
        ├── classe1/
        │   └── imagem5.jpg
        └── classe2/
            └── imagem6.jpg
```

2. Executar o Sistema

```Shell
streamlit run main.py
```

3. Fluxo de Trabalho

   - Selecione o Dataset na barra lateral
   - Escolha o Descritor (Histograma, HOG, LBP ou SIFT)
   - Escolha o Classificador (KNN, SVM, MLP ou Random Forest)
   - Extraia as Características na aba correspondente
   - Treine o Modelo na aba de treinamento
   - Avalie os Resultados na aba de resultados

## 🔄 Pipeline do Sistema

```
┌─────────────────────────────────────────────────────────────────┐
│                         PIPELINE DO SISTEMA                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌────────────────┐    ┌──────────────────┐  │
│  │   Imagens   │    │  Extração de   │    │  Características │  │
│  │  (Dataset)  │───▶│ Características│───▶│   (Features)     │  │
│  └─────────────┘    └────────────────┘    └──────────────────┘  │
│         │                  │                      │             │
│         │                  │                      │             │
│         ▼                  ▼                      ▼             │
│  ┌─────────────┐    ┌────────────────┐    ┌──────────────────┐  │
│  │  Treinamento│    │  Avaliação     │    │   Resultados     │  │
│  │  (Modelo)   │◀───│ (Classificação)│─── │  (Métricas)      │  │
│  └─────────────┘    └────────────────┘    └──────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 📊 Resultados

Os resultados são salvos automaticamente na pasta resultados/ com a seguinte estrutura:

```
resultados/[Dataset]/[Descritor]/[Classificador]/
├── matriz_confusao.png                           # Matriz de confusão
└── relatorio_classificacao.png                   # Métricas detalhadas
```

As métricas calculadas incluem:

- Accuracy: Acurácia geral do modelo
- Precision: Precisão por classe
- Recall: Recall por classe
- F1-Score: F1-Score por classe

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo LICENSE para mais detalhes.

## 👨‍💻 Autor

  João Guilherme - Desenvolvedor IA -  [github.com/Joao-gui](https://github.com/Joao-gui?)

⭐ Se este projeto foi útil para você, considere dar uma estrela! ⭐
