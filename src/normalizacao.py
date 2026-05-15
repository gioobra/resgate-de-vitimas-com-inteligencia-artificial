import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

class NormalizadorSinaisVitais:
    def __init__(self, nome_arquivo=None):
        self.file_path = None
        if nome_arquivo:
            self.file_path = Path(__file__).parent.parent / 'data' / nome_arquivo
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def carregar_dados(self, path=None):
        target_path = Path(path) if path else self.file_path
        df = pd.read_csv(target_path, header=None)
        # Assumindo que a estrutura é (ID, Feat1, ..., FeatN, Label?)
        # Se for o dataset de treino (com label), ignora primeira e última
        # Se for teste, ignora apenas a primeira.
        # Vamos tratar no modelos.py a seleção de colunas.
        return df

    def processar(self, feature_range=(0, 1)):
        """Método original mantido para compatibilidade."""
        if not self.file_path:
            raise ValueError("Caminho do arquivo não definido.")
        df = pd.read_csv(self.file_path, header=None)
        X = df.iloc[:, 1:-1]
        y = df.iloc[:, -1]
        self.scaler = MinMaxScaler(feature_range=feature_range)
        X_normalizado = self.scaler(X)
        return pd.DataFrame(X_normalizado), y
