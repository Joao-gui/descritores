# ==========================================================
# SISTEMA DE CLASSIFICAÇÃO DE IMAGENS
# Streamlit
# ==========================================================

import os
import time
import streamlit as st
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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

    if "tempo_treinamento" not in st.session_state:
        st.session_state.tempo_treinamento = None

    # ==========================================================
    # ABAS
    # ==========================================================

    tab_dataset, tab_caracteristicas, tab_treinamento, tab_resultados = st.tabs(
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
        st.sidebar.error("O dataset deve possuir as pastas 'treino' e 'teste'.")

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
                width='stretch'
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
            width='stretch'
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
                    caracteristicas_treino, pca_ajustado = (
                        histograma.extrai_histograma_escala_cinza(imagens_treino)
                    )

                    caracteristicas_teste, _ = (
                        histograma.extrai_histograma_escala_cinza(imagens_teste, pca=pca_ajustado)
                    )

                elif descritor == "HOG":
                    caracteristicas_treino, pca_ajustado = (
                        hog.extrair_hog(imagens_treino)
                    )

                    caracteristicas_teste, _ = (
                        hog.extrair_hog(imagens_teste, pca=pca_ajustado)
                    )

                elif descritor == "LBP":
                    caracteristicas_treino, pca_ajustado = (
                        lbp.extrair_lbp(imagens_treino)
                    )

                    caracteristicas_teste, _ = (
                        lbp.extrair_lbp(imagens_teste, pca=pca_ajustado)
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

    # ==========================================================
    # ABA - TREINAMENTO
    # ==========================================================

    caracteristicas_treino = st.session_state.caracteristicas_treino
    caracteristicas_teste = st.session_state.caracteristicas_teste

    rotulos_treino = st.session_state.rotulos_treino
    rotulos_teste = st.session_state.rotulos_teste

    with tab_treinamento:

        st.header("🤖 Treinamento do Modelo")

        if caracteristicas_treino is None:
            st.warning("Primeiro extraia as características.")

        elif rotulos_treino is None:
            st.warning("Rótulos de treinamento não encontrados.")

        else:
            st.write(f"**Descritor:** {st.session_state.descritor}")
            st.write(f"**Classificador:** {classificador}")

            st.divider()

            treinar = st.button(
                "🚀 Treinar Modelo",
                width='stretch'
            )

            if treinar:
                inicio = time.time()

                with st.spinner("Treinando modelo..."):

                    # ======================
                    # KNN
                    # ======================

                    if classificador == "KNN":
                        modelo = knn.treinar_knn(
                            caracteristicas_treino,
                            rotulos_treino
                        )

                    # ======================
                    # SVM
                    # ======================

                    elif classificador == "SVM":
                        modelo = svm.treinar_svm(
                            caracteristicas_treino,
                            rotulos_treino
                        )

                    # ======================
                    # MLP
                    # ======================

                    elif classificador == "MLP":
                        modelo = mlp.treinar_mlp(
                            caracteristicas_treino,
                            rotulos_treino
                        )

                    # ======================
                    # Random Forest
                    # ======================

                    elif classificador == "Random Forest":
                        modelo = random_forest.treinar_rf(
                            caracteristicas_treino,
                            rotulos_treino
                        )

                tempo = round(time.time() - inicio, 2)


                # Guardar modelo

                st.session_state.modelo = modelo
                st.session_state.classificador = classificador
                st.session_state.tempo_treinamento = tempo

                # Salvar modelo noo disco
                CAMINHO_MODELO = (f"modelos/{classificador}/modelo.pkl")

                dados.salvar_modelo(modelo, CAMINHO_MODELO)

                st.success("Modelo treinado e salvo com sucesso!")
                st.metric("Tempo de treinamento", f"{tempo}")
                st.write(f"Modelo salvo em: `{CAMINHO_MODELO}`")

    # ==========================================================
    # ABA - RESULTADO
    # ==========================================================

    with tab_resultados:

        st.header("📊 Resultado")

        modelo = st.session_state.modelo
        caracteristicas_teste = st.session_state.caracteristicas_teste
        rotulos_teste = st.session_state.rotulos_teste
        encoder = st.session_state.encoder

        if modelo is None:
            st.warning("Treine um modelo antes de realizar a classificação.")

        else:
            classificador_treinado = st.session_state.classificador
            
            st.write(f"**Descritores:** {st.session_state.descritor}")
            st.write(f"**Classificador:** {classificador_treinado}")

            if classificador != classificador_treinado:
                st.warning(
                    f"O menu lateral está em **{classificador}**, mas o modelo em memória "
                    f"foi treinado com **{classificador_treinado}**. Volte na aba "
                    "'🤖 Treinamento' e clique em 'Treinar Modelo' para treinar com o "
                    "classificador selecionado antes de gerar o resultado."
                )

            if st.session_state.tempo_treinamento is not None:
                st.write(f"**Tempo de treinamento:** {st.session_state.tempo_treinamento:.2f}s")

            st.divider()

            prever = st.button("🚀 Realizar Classificação", width='stretch')

            if prever:
                with st.spinner("Realizando classificação..."):

                    # ==========================================
                    # KNN
                    # ==========================================
                    if classificador_treinado == "KNN":
                        rotulos_previstos = knn.testar_knn(modelo, caracteristicas_teste)

                    # ==========================================
                    # SVM
                    # ==========================================
                    elif classificador_treinado == "SVM":
                        rotulos_previstos = svm.testar_svm(modelo, caracteristicas_teste)

                    # ==========================================
                    # MLP
                    # ==========================================
                    elif classificador_treinado == "MLP":
                        rotulos_previstos = mlp.testar_mlp(modelo, caracteristicas_teste)

                    # ==========================================
                    # RANDOM FEOREST
                    # ==========================================
                    elif classificador_treinado == "Random Forest":
                        rotulos_previstos = random_forest.testar_rf(modelo, caracteristicas_teste)

                    st.session_state.rotulos_previstos = rotulos_previstos

                st.success("Classificação realizada com sucesso!")

                # ==================================================
                # MÉTRICAS (Accuracy, precision, recall e f1)
                # ==================================================

                accuracy = accuracy_score(rotulos_teste, rotulos_previstos)
                precision = precision_score(rotulos_teste, rotulos_previstos, average='weighted')
                recall = recall_score(rotulos_teste, rotulos_previstos, average='weighted')
                f1 = f1_score(rotulos_teste, rotulos_previstos, average='weighted')

                # ==================================================
                # CAMINHOS
                # ==================================================
                nomes_classes = encoder.classes_

                CAMINHO_RESULTADOS = os.path.join(
                    "resultados",
                    dataset,
                    st.session_state.descritor,
                    classificador_treinado.replace(" ", "")
                )

                os.makedirs(CAMINHO_RESULTADOS, exist_ok=True)

                caminho_matriz = os.path.join(
                    CAMINHO_RESULTADOS,
                    "matriz_confusao.png"
                )

                caminho_relatorio = os.path.join(
                    CAMINHO_RESULTADOS,
                    "realtorio_calssificacao.png"
                )

                metricas.matriz_confusao(
                    nomes_das_classes=nomes_classes,
                    rotulos_verdadeiros=rotulos_teste,
                    rotulos_previstos=rotulos_previstos,
                    caminho_arquivo=caminho_matriz
                )

                metricas.relatorio_classificacao(
                    nomes_das_classes=nomes_classes,
                    rotulos_verdadeiros=rotulos_teste,
                    rotulos_previstos=rotulos_previstos,
                    caminho_arquivo=caminho_relatorio,
                )

                st.divider()
                
                # ==================================================
                # MATRIZ DE CONFUSÃO + MÉTRICAS
                # ==================================================
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Matriz Confusão")

                    st.image(caminho_matriz, width=500) #width="stretch"

                with col2:
                    st.subheader("Métricas")

                    st.metric("Accuracy", f"{accuracy*100:.2f}%")
                    st.metric("Precision", f"{precision*100:.2f}%")
                    st.metric("Recall", f"{recall*100:.2f}%")
                    st.metric("F1-Score", f"{f1*100:.2f}%")

                st.divider()

                # ==================================================
                # RELATÓRIO DE CLASSIFICAÇÃO
                # ==================================================
                st.subheader("Relatório de Classificação")

                st.image(caminho_relatorio, width=800)
                
                st.success("Resultado gerado com sucesso!")

                st.write("Arquivos salvos em:")

                st.code(CAMINHO_RESULTADOS)