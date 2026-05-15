import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

class NormalizadorSinaisVitais:
    def __init__(self, nome_arquivo):
        self.file_path = Path(__file__).parent.parent / 'data' / nome_arquivo

    def carregar_dados(self):
        df = pd.read_csv(self.file_path, header=None)
        # Pega todas as linhas (:), ignorando a primeira e a última coluna 
        X = df.iloc[:, 1:-1]
        # Pega todas as linhas (:), mas apenas a última coluna 
        y = df.iloc[:, -1]
        return X, y

    def processar(self, feature_range=(0, 1)):
        X, y = self.carregar_dados()
        # Aplica a normalização Min-Max 
        scaler = MinMaxScaler(feature_range=feature_range)
        X_normalizado = scaler.fit_transform(X)
        return pd.DataFrame(X_normalizado), y