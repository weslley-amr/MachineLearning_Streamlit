import streamlit as st
import pandas as pd
import joblib
import os

#nome_usuario = st.text_input("Informe o seu nome: ", placeholder="Digite aqui... ")

#if st.button(label="Clique aqui"):
    #st.write(f"Seja bem-vindo,", nome_usuario)

# 1. Definição de features
FEATURE_NAMES =[
    "Nota_P1",
    "Nota_P2",
    "Media_Trabalhos",
    "Frequencia",
    "Reprovacoes_Anteriores",
    "Acessos_Plataforma_Mes"
]

COLUNAS_HISTORICO = FEATURE_NAMES + ["Previsao_Resultado", "Prob_Aprovado", "Prob_Reprovado"]

#Criar uma sessão st.session
if "historico_previsoes" not in st.session_state:
    st.session_state.historico_previsoes = pd.DataFrame(columns=COLUNAS_HISTORICO)

# 2. Carregamento do modelo para o nosso front-end
#st.cache_resource para carregar o modelo apenas uma vez
#otimizando o desempenho do app
@st.cache_resource
def carregar_modelo(caminho_modelo = "modelo_previsao_desempenho.joblib"):
    """
    Carregar o pipeline de ML treinado (scaler + modelo) do arquivo .joblib
    """

    try:
        if os.path.exists(caminho_modelo):
            modelo = joblib.load(caminho_modelo)
            return modelo
        else:
            st.error(f"Erro: Arquivo do modelo '{caminho_modelo}' não foi encontrado!")
            st.warning(f"Por favor, execute o script 'modelo_treinamento.py' para gerar o modelo")
    except Exception as e:
        st.error(f"Erro inesperado ao carregar o modelo: {e}")
        return None
    
pipeline_modelo= carregar_modelo()

# 2. Configuração da interface do usuário(streamlit)
st.set_page_config(layout="wide", page_title="Previsão de notas")

st.title("🦍 Previsor de desempenho academico")
st.markdown( """ 
    Essa ferramenta usa Inteligencia Artificial para prever o status final de um aluno
    com base em seu desempenho parcial

    **Preencha os dados do aluno abaixo para obter um aprevisão: **
""")

# 3. Formulário de entrada

if pipeline_modelo is not None:
    #Utilizar um formulário para grupar as entradas e o botão
    with st.form("Formulário de previsão"):
        st.subheader("Insira as notas e métricas dos alunos")

        col1, col2 = st.columns(2)

        with col1:
            nota_p1 = st.slider("Nota da 1° prova (0 a 10)", min_value=0.0, max_value=10.0, value=5.0, step=0.5)
            media_trabalhos = st.slider("Média do trabalhos (0 a 10)", min_value=0.0, max_value=10.0, value=5.0, step=0.5)
            reprovacoes_anteriores = st.number_input("Reprovações anteriores", min_value=0, max_value=10, value=5, step=1)
        with col2:
            nota_p2 = st.slider("Nota da 2° prova (0 a 10)", min_value=0.0, max_value=10.0, value=5.0, step=0.5)
            frequencia = st.slider("Frequência (%)", min_value=0, max_value=100, value=75, step=5)
            acesso_mes = st.number_input("Média de acessos a plataforma (por mês)", min_value=0, max_value=100, value=10, step=1)

        submitted = st.form_submit_button("Realizar previsão")

    if submitted:

        features_name =[
            "ï»¿Nota_P1",
            "Nota_P2",
            "Media_Trabalhos",
            "Frequencia",
            "Reprovacoes_Anteriores",
            "Acessos_Plataforma_Mes"
        ]

    #Criação de dataframe a partir dos dados inseridos
        dados_alunos = pd.DataFrame(
          [[nota_p1, nota_p2, media_trabalhos, frequencia, reprovacoes_anteriores, acesso_mes]],
          columns=features_name
    )
        st.info("Processando dados e realizando a previsão...")

        try:
            #Realizar a previsão ([0] ou [1])
            previsao = pipeline_modelo.predict(dados_alunos)

            #Obter a probabilidade 
            probabilidade= pipeline_modelo.predict_proba(dados_alunos)

            prob_reprovados = probabilidade[0][0]
            prob_aprovados = probabilidade [0][1]
            resultado_texto = "APROVADO!" if previsao[0] == 1 else "REPROVADO!"

            #Exibir os resultados na tela

            st.subheader("Resultado da previsão")

            if previsao[0] == 1:
                st.success("Previsão: Aprovado")
                st.markdown(f""" 
                    Com base nos dados fornecidos, o modelo prevê que o aluno tem:
                    **{prob_aprovados*100:.2f}%** de chace de ser **aprovado**

                    *Chance de de reprovação: {prob_reprovados*100:.2f}%*
                """)
            else:
                st.error("Previsão: Reprovado")
                st.markdown(f""" 
                    Com base nos dados fornecidos, o modelo prevê que o aluno tem:
                    **{prob_reprovados*100:.2f}%** de chace de ser **reprovado**

                    *Chance de de aprovação: {prob_aprovados*100:.2f}%*
                """)

            nova_linha_dict= {
                "Nota_P1": nota_p1,
                "Nota_P2": nota_p2,
                "Media_Trabalhos": media_trabalhos,
                "Frequencia": frequencia,
                "Reprovacoes_Anteriores": reprovacoes_anteriores,
                "Acessos_Plataforma_Mes": acesso_mes,
                "Previsao_Resultado": resultado_texto,
                "Prob_Aprovado": round(prob_aprovados *100,2),
                "Prob_Reprovado": round(prob_reprovados*100,2)
            }

            nova_linha_df = pd.DataFrame([nova_linha_dict], columns=COLUNAS_HISTORICO)

            st.session_state.historico_previsoes = pd.concat(
                [st.session_state.historico_previsoes, nova_linha_df],
                ignore_index=True
            )

        except Exception as e:
            st.error(f"Erro ao fazer a previsão: {e}")
            st.error("Verifique se os nomes das colunas correspondem aos nomes das colunas utilizadas no treino")

    st.subheader("Histórico de previsões realizadas na sessão: ")
    if st.session_state.historico_previsoes.empty:
        st.write("Nenhuma previsão foi realizada ainda")
    else:
        st.dataframe(st.session_state.historico_previsoes, use_container_width=True)

        if st.button("Limpar histórico"):
            st.session_state.historico_previsoes= pd.DataFrame(columns= COLUNAS_HISTORICO)

            st.rerun()

else:
    st.warning("O aplicativo não pode fazer previsões porque o modelo não foi carregado!")
        