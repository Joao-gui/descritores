# ==========================================================
# SISTEMA DE CLASSIFICAÇÃO DE IMAGENS
# Streamlit
# ==========================================================

import os
import time
import streamlit as st
import pandas as pd

# ==========================================================
# Módulos do projeto
# ==========================================================

from modulos.utils import dados, metricas, rotulos
from modulos.descritores import histograma, hog, lbp, sift
from modulos.classificadores import knn, mlp, random_forest, svm

def run():
    # ==========================================================
    # CONFIGURAÇÃOO DA PÁGINA
    # ==========================================================

    st.set_page_config(
        page_title='Sistema de Classificação de Imagens',
        page_icon='🖼️',
        layout='wide',
        initial_sidebar_state='expanded'
    )

    # ==========================================================
    # TÍTULO
    # ==========================================================

    st.title('Sistema de Classificação de Imagens')

    st.markdown("""
    Este sistema permite comparar diferentes **descritores de imagens**
    e diferentes **algoritmos de classificação**.

    Projeto desenvolvido para a disciplina de Visão Computacional.
    """)

    # ==========================================================
    # SEASSIN STATE (Evita que o streamlit perca dados (como o 
    # modelo treinado) sempre que o usuário altera um componente)
    # ==========================================================

    if "imagens_treino" not in st.session_state:
        st.session_state.imagens_treino = None

    if "rotulos_treino" not in st.session_state:
        st.session_state.rotulos_treino = None

    if "imagens_teste" not in st.session_state:
        st.session_state.imagens_teste = None

    if "rotulos_teste" not in st.session_state:
        st.session_state.rotulos_teste = None

    if "caracteristicas_treino" not in st.session_state:
        st.session_state.caracteristicas_treino = None

    if "caracteristicas_teste" not in st.session_state:
        st.session_state.caracteristicas_teste = None

    if "modelo" not in st.session_state:
        st.session_state.modelo = None

    if "encoder" not in st.session_state:
        st.session_state.encoder = None

    if "rotulos_previstos" not in st.session_state:
        st.session_state.rotulos_previstos = None

    if "descritor" not in st.session_state:
        st.session_state.descritor = None

    if "classificador" not in st.session_state:
        st.session_state.classificador = None

    if "modelo_kmeans" not in st.session_state:
        st.session_state.modelo_kmeans = None

    if "n_grupos" not in st.session_state:
        st.session_state.n_grupos = None

    # ==========================================================
    # ABAS
    # ==========================================================

    tab_dataset, tab_caracteristicas, tab_treinamento, tab_resultadoos = st.tabs(
        [
            "📂 Dataset",
            "🧬 Características",
            "🤖 Treinamento",
            "📊 Resultado"
        ]
    )

    # ==========================================================
    # DIRETÓRIOS
    # ==========================================================

    DIRETORIO_DATASETS = 'datasets'

    # ==========================================================
    # LISTAR DATASETS
    # ==========================================================

    def listar_datasets():
        if not os.path.exists(DIRETORIO_DATASETS):
            return []

        datasets = []

        for pasta in os.listdir(DIRETORIO_DATASETS):
            caminho = os.path.join(DIRETORIO_DATASETS, pasta)

            if os.path.isdir(caminho):
                datasets.append(pasta)

        return sorted(datasets)

    # ==========================================================
    # VALIDAR DATASET
    # ==========================================================

    def validar_dataset(nome_dataset):
        caminho_dataset = os.path.join(DIRETORIO_DATASETS, nome_dataset)
        treino = os.path.join(caminho_dataset, "train")
        teste = os.path.join(caminho_dataset, "test")

        if not os.path.exists(treino):
            return False

        if not os.path.exists(teste):
            return False

        return True

    # ==========================================================
    # INFORMAÇÕES DO DATASET
    # ==========================================================

    def obter_informacoes_dataset(nome_dataset):
        caminho_dataset = os.path.join(DIRETORIO_DATASETS, nome_dataset)
        caminho_treino = os.path.join(caminho_dataset, "train")
        caminho_teste = os.path.join(caminho_dataset, "test")

        classes = []

        imagens_treino = 0
        imagens_teste = 0

        # --------
        # TREINO
        # --------
        if os.path.exists(caminho_treino):
            for classe in sorted(os.listdir(caminho_treino)):
                caminho_classe = os.path.join(caminho_treino, classe)
                if os.path.isdir(caminho_classe):
                    classes.append(classe)
                    imagens_treino += len(
                        [
                            arquivo for arquivo in os.listdir(caminho_classe)
                            if os.path.isfile(os.path.join(caminho_classe, arquivo))
                        ]
                    )

        # -------
        # TESTE
        # -------
        if os.path.exists(caminho_teste):
            for classe in sorted(os.listdir(caminho_teste)):
                caminho_classe = os.path.join(caminho_teste, classe)
                if os.path.isdir(caminho_classe):
                    imagens_teste += len(
                        [
                            arquivo for arquivo in os.listdir(caminho_classe)
                            if os.path.isfile(os.path.join(caminho_classe, arquivo))
                        ]
                    )

        return{
            "classes": classes,
            "numero_classes": len(classes),
            "treino": imagens_treino,
            "teste": imagens_teste,
            "total": imagens_treino + imagens_teste
        }

    # ==========================================================
    # SIDEBAR
    # ==========================================================

    st.sidebar.title("Configurações")

    lista_datasets = listar_datasets()

    dataset = st.sidebar.selectbox(
        "Dataset",
        lista_datasets
    )

    caracteristicas_treino = None
    caracteristicas_teste = None
    modelo_kmeans = None
    n_grupos = None

    descritor = st.sidebar.selectbox(
        "Descritor",
        [
            "Histograma",
            "HOG",
            "LBP",
            "SIFT"
        ]
    )

    classificador = st.sidebar.selectbox(
        "Classificador",
        [
            "KNN",
            "SVM",
            "MLP",
            "Random Forest"
        ]
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Resumo")
    st.sidebar.write(f"**Dataset:** {dataset}")
    st.sidebar.write(f"**Descritor:** {descritor}")
    st.sidebar.write(f"**Classificador:** {classificador}")

    if validar_dataset(dataset):
        st.sidebar.success("Dataset válido")
        #info = obter_informacoes_dataset(dataset)
        #st.sidebar.markdown('---')
        #st.sidebar.subheader("Informações")
        #st.sidebar.write(f"Classes: {info['numero_classes']}")
        #st.sidebar.write(f"Treino: {info['treino']} imagens")
        #st.sidebar.write(f"Teste: {info['teste']} imagens")
        #st.sidebar.write(f"Total: {info['total']} imagens")

    else:
        st.sidebar.error(
            "O dataset deve possuir as pastas 'treino' e 'teste'."
        )

    # ==========================================================
    # ABA - DATASET
    # ==========================================================

    with tab_dataset:
        st.header("📂 Informações do Dataset")

        if validar_dataset(dataset):
            info = obter_informacoes_dataset(dataset)

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Classes",
                    info['numero_classes']
                )

            with col2:
                st.metric(
                    "Treino",
                    info['treino']
                )

            with col3:
                st.metric(
                    "Teste",
                    info['teste']
                )

            st.metric(
                "Total de imagens",
                info['total']
            )

            st.subheader("Classes encontradas")

            df = pd.DataFrame(
                info['classes'],
                columns=["Classe"]
            )

            st.dataframe(
                df,
                use_container_width=True
            )

        else:
                st.error("Dataset inválido.")

    # ==========================================================
    # ABA - EXTRAÇÃO DE CARACTERÍSTICAS
    # ==========================================================

    with tab_caracteristicas:
        st.header("🧬 Extração de Características")

        st.write(
            "Extraia as características das imgens utilizando "
            "o descritor selecionado."
        )

        st.divider()

        extrair = st.button(
            "🚀 Extrair Características",
            use_container_width=True
        )

        # Quando clicar
        if extrair:
            caminho_dataset = os.path.join(
                DIRETORIO_DATASETS,
                dataset
            )

            caminho_treino = os.path.join(
                caminho_dataset,
                "train"
            )

            caminho_teste = os.path.join(
                caminho_dataset,
                "test"
            )

            # Barra de progresso
            with st.spinner("Carregando imagens..."):
                # Carregaar Treino
                imagens_treino, rotulos_treino = dados.carregar_imagens(caminho_treino)

                # Carregar Teste
                imagens_teste, rotulos_teste = dados.carregar_imagens(caminho_teste)

                # Guardar no session state
                st.session_state.imagens_treino = imagens_treino
                st.session_state.rotulos_treino = rotulos_treino
                st.session_state.imagens_teste = imagens_teste
                st.session_state.rotulos_teste = rotulos_teste

                # Codificar rótulos
                rotulos_codificados_treino, encoder = (
                    rotulos.codificar_rotulos_label(rotulos_treino)
                )

                # Transform com LabelEncoder
                rotulos_codificados_teste = encoder.transform(rotulos_teste)

                # Guardar no session state
                st.session_state.encoder = encoder
                st.session_state.rotulos_treino = rotulos_codificados_treino
                st.session_state.rotulos_teste = rotulos_codificados_teste
                st.session_state.rotulos_originais_treino = rotulos_treino
                st.session_state.rotulos_originais_teste = rotulos_teste

                # Escolher Descritor
                if descritor == "Histograma":
                    caracteristicas_treino = (
                        histograma.extrai_histograma_escala_cinza(imagens_treino)
                    )

                    caracteristicas_teste = (
                        histograma.extrai_histograma_escala_cinza(imagens_teste)
                    )

                elif descritor == "HOG":
                    caracteristicas_treino = (
                        hog.extrair_hog(imagens_treino)
                    )

                    caracteristicas_teste = (
                        hog.extrair_hog(imagens_teste)
                    )

                elif descritor == "LBP":
                    caracteristicas_treino = (
                        lbp.extrair_lbp(imagens_treino)
                    )

                    caracteristicas_teste = (
                        lbp.extrair_lbp(imagens_teste)
                    )

                elif descritor == "SIFT":
                    # Treino
                    (caracteristicas_treino, modelo_kmeans, n_grupos, rotulos_codificados_treino) = sift.extrai_sift_treinamento(imagens_treino, rotulos_codificados_treino)
                    #Teste
                    (caracteristicas_teste, rotulos_codificados_teste) = sift.extrai_sift_teste(imagens_teste, modelo_kmeans, n_grupos, rotulos_codificados_teste)

                # Guardar no session state
                st.session_state.caracteristicas_treino = caracteristicas_treino
                st.session_state.caracteristicas_teste = caracteristicas_teste
                st.session_state.descritor = descritor
                st.session_state.modelo_kmeans = modelo_kmeans
                st.session_state.n_grupos = n_grupos

                # Salvar no disco
                CAMINHO_CARACTERISTICAS_DIR = "features/caracteristicas/" + descritor
                dados.salvar_caracteristicas(
                    caracteristicas_treino, CAMINHO_CARACTERISTICAS_DIR + "/treino.pkl"
                )

                dados.salvar_caracteristicas(
                    caracteristicas_teste, CAMINHO_CARACTERISTICAS_DIR + "/teste.pkl"
                )

                # Resumo
                st.success("Caracteristicas extraidas com sucesso!")

                col1, col2 = st.columns(2)

                with col1:
                    st.metric(
                        "Imagens de Treino",
                        len(caracteristicas_treino)
                    )

                with col2:
                    st.metric(
                        "Imagens de Teste",
                        len(caracteristicas_teste)
                    )

                st.write("Formato das caracteristicas:")


                st.code(
                    f"Treino: {caracteristicas_treino.shape}\n"
                    f"Teste: {caracteristicas_teste.shape}"
                )

                st.write(f"**Arquivos salvos em:** {CAMINHO_CARACTERISTICAS_DIR}")