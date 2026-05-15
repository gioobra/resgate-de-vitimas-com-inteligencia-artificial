import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

class ClassificadorSinaisVitais:
    def __init__(self, arquivo_treino='02_treino_sinais_vitais_com_label.txt', arquivo_teste='01_teste_sinais_vitais_sem_label.txt'):
        self.arquivo_treino = arquivo_treino
        self.arquivo_teste = arquivo_teste
        self.modelo = RandomForestClassifier(n_estimators=100, random_state=42)

    def carregar_dados(self, nome_arquivo, possui_label=True):
        file_path = Path(__file__).parent.parent / 'data' / nome_arquivo
        df = pd.read_csv(file_path, header=None)
        
        if possui_label:
            X = df.iloc[:, 1:-1]
            y = df.iloc[:, -1]
        else:
            X = df.iloc[:, 1:]
            y = None
        return X, y

    def treinar_e_avaliar(self):
        # Preparar dados de treino (Documento 02)
        X_train, y_train = self.carregar_dados(self.arquivo_treino, possui_label=True)
        
        self.modelo.fit(X_train, y_train)
        
        # Preparar dados onde o modelo será utilizado (Documento 01)
        possui_label_teste = 'sem_label' not in self.arquivo_teste
        X_test, y_test = self.carregar_dados(self.arquivo_teste, possui_label=possui_label_teste)
        
        y_pred = self.modelo.predict(X_test)

        if y_test is not None:
            accuracy = accuracy_score(y_test, y_pred)
            classification_rep = classification_report(y_test, y_pred)
            print(f"Accuracy: {accuracy:.2f}")
            print("\nClassification Report:\n", classification_rep)
        else:
            print("Arquivo de teste não possui labels. Predições realizadas com sucesso, mas métricas ignoradas.")
        
        return self.modelo, y_test, y_pred

if __name__ == "__main__":
    classificador_rf = ClassificadorSinaisVitais()
    modelo_rf, y_test_rf, y_pred_rf = classificador_rf.treinar_e_avaliar()
    
