import pandas as pd
from pathlib import Path

class DataLoader:
    def __init__(self, data_dir=None):
        if data_dir:
            self.data_dir = Path(data_dir)
        else:
            self.data_dir = Path(__file__).parent.parent / 'data'

    def carregar_dados_treino(self, nome_arquivo='02_treino_sinais_vitais_com_label.txt'):
        """Carrega o dataset com labels para treinamento."""
        path = self.data_dir / nome_arquivo
        df = pd.read_csv(path, header=None)
        # Primeira coluna é ID, última é label. Features estão entre elas.
        X = df.iloc[:, 1:-1]
        y = df.iloc[:, -1]
        return X, y

    def carregar_dados_teste(self, nome_arquivo='01_treino_sinais_vitais_sem_label.txt'):
        """Carrega o dataset sem labels para teste/inferência."""
        path = self.data_dir / nome_arquivo
        df = pd.read_csv(path, header=None)
        # Primeira coluna é ID. Resto são features (não há label).
        X = df.iloc[:, 1:]
        return X
