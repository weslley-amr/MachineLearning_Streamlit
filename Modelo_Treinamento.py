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
    
# Chamar a função para armazenar o resultado

dados= carregar_dados()
print(dados)