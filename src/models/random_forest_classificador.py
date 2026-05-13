import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import MinMaxScaler

class ClassificadorSinaisVitais:
    def __init__(
        self,
        arquivo_treino='02_treino_sinais_vitais_com_label.txt',
        arquivo_classificacao='01_treino_sinais_vitais_sem_label.txt',
    ):
        self.data_dir = Path(__file__).parent.parent / 'data'
        self.arquivo_treino = self.data_dir / arquivo_treino
        self.arquivo_classificacao = self.data_dir / arquivo_classificacao
        self.modelo = RandomForestClassifier(n_estimators=100, random_state=42)
        self.normalizador = MinMaxScaler(feature_range=(0, 1))

    def _carregar_treino_com_label(self):
        df_treino = pd.read_csv(self.arquivo_treino, header=None)
        ids = df_treino.iloc[:, 0]
        X_treino = df_treino.iloc[:, 1:-1]
        y_treino = df_treino.iloc[:, -1]
        return ids, X_treino, y_treino

    def _carregar_amostras_sem_label(self):
        df_sem_label = pd.read_csv(self.arquivo_classificacao, header=None)
        ids = df_sem_label.iloc[:, 0]
        X_sem_label = df_sem_label.iloc[:, 1:]
        return ids, X_sem_label

    def treinar_e_avaliar(self):
        ids_treino, X_treino, y_treino = self._carregar_treino_com_label()
        ids_sem_label, X_sem_label = self._carregar_amostras_sem_label()

        X_treino_norm = self.normalizador.fit_transform(X_treino)
        X_sem_label_norm = self.normalizador.transform(X_sem_label)

        self.modelo.fit(X_treino_norm, y_treino)
        y_pred_sem_label = self.modelo.predict(X_sem_label_norm)

        referencia = pd.DataFrame({'id': ids_treino, 'label_real': y_treino})
        previsoes = pd.DataFrame({'id': ids_sem_label, 'label_predito': y_pred_sem_label})
        avaliacao = previsoes.merge(referencia, on='id', how='left')

        if avaliacao['label_real'].isna().any():
            raise ValueError('Existem IDs no arquivo sem label que nao foram encontrados na base com label.')

        accuracy = accuracy_score(avaliacao['label_real'], avaliacao['label_predito'])
        classification_rep = classification_report(avaliacao['label_real'], avaliacao['label_predito'])

        print(f"Accuracy: {accuracy:.2f}")
        print("\nClassification Report:\n", classification_rep)

        return self.modelo, avaliacao['label_real'], avaliacao['label_predito']

if __name__ == "__main__":
    classificador_cart = ClassificadorSinaisVitais()
    modelo_cart, y_test_cart, y_pred_cart = classificador_cart.treinar_e_avaliar()
    
