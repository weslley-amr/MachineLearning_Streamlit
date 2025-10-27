import pandas as pd
import joblib
import os
from sklearn import model_selection, preprocessing, pipeline, linear_model, metrics

# 1. Carregar dados
def carregar_dados(caminho_arquivo = "historicoAcademico.csv"):
    try:
        #Carregamento dos dados
        if os.path.exists(caminho_arquivo):

            df= pd.read_csv(caminho_arquivo, encoding="latin1", sep=",")
            print("O arquivo foi carregado com sucesso!")
            return df
        else:
            print("O arquivo não foi encontrado dentro da pasta!")

            return None
    except Exception as e:
        print("Erro inesperado ao carregar o arquivo: ", e)

        return None
    
#Chamar a função para armazenar o resultado

dados= carregar_dados()

# 2. Preparação e divisão dos dados
#Definição de X (features) e Y (target)

if dados is not None:
    print(f"\nTotal de reguistros carregados: {len(dados)}")
    print("Iniciando o pipeline de treinamento")

    TARGET_COLUMN = "Status_Final"

    #Definição das features e target
    try:
        X = dados.drop(TARGET_COLUMN, axis=1)
        Y = dados[TARGET_COLUMN]

        print(f"Feature (X) definidas: {list(X.columns)}")
        print(f"Feature (Y) definidas: {TARGET_COLUMN}")

    except KeyError:

        print(f"\n ----- Erro Crítico -----")
        print(f"A coluna {TARGET_COLUMN} não foi encontrada no CSV")
        print(f"Colunas disponíveis: {list(dados.columns)}")
        print(f"Por favor, ajuste a váriavel 'TARGET_COLUMN' e tente novamente!")
        #Se o target não for encontrado, irá encerrar o script!
        exit()

    #Divisão entre treino e teste
    print("\n ------- Dividindo dados em treino e teste... ------")

    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(
        X, Y,
        test_size=0.2,      #20% dos dados serão utilizados para teste
        random_state= 42,   #Garantir a reprodutibilidade
        stratify=Y          #Manter a proporção de aprovados e reprovados
    )

    print(f"Dados de treino: {len(X_train)} | Dados de teste: {len(X_test)}")

 # 3. Criação da pipeline de machinelearning
    print("\n------ Criando a pipeline de machinelearning... ------")
    #scaler -> normalização dos dados (colocando tudo na mesma escala)
    #model -> aplica o modelo de regressão logística
    pipeline_model = pipeline.Pipeline([
        ("scaler", preprocessing.StandardScaler()),
        ("model", linear_model.LogisticRegression(random_state=42))
    ])

# 4. Treinamento e avaliação dos daods/modelo
    print("\n ------ Treinamneto do modelo... ------")
    #Treina a pipeline com os dados de treino
    pipeline_model.fit(X_train, Y_train)

    print("modelo treinado. Avaliando com os dados de teste...")
    Y_pred = pipeline_model.predict(X_test)

    #Avaliação de desempenho
    accuracy = metrics.accuracy_score(Y_test, Y_pred)
    report =metrics.classification_report(Y_test, Y_pred)

    print("\n ------ Relatório de avaliação geral ------")
    print(f"Precisão Geral: {accuracy * 100:.2f}%")
    print("\nRelatório de classificação detalhado:")
    print(report)

# 5. Salvando o modelo
    model_filename = "modelo_previsao_desempenho.joblib"

    print(f"\nSalvando o pipeline treinado em... {model_filename}")
    joblib.dump(pipeline_model, model_filename)

    print("Processo concluído com sucesso")
    print(f"O arquivo '{model_filename}' está pronto para ser utilizado!")

else:
    print("O pipeline não pode continuar pois os dados não foram carregados!")