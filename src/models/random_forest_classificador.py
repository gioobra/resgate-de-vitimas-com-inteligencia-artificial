import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import MinMaxScaler

class ClassificadorSinaisVitais:
    def __init__(self, arquivo_dados='02_treino_sinais_vitais_com_label.txt'):
        self.arquivo_dados = arquivo_dados
        self.normalizador = NormalizadorSinaisVitais(self.arquivo_dados)
        self.modelo = RandomForestClassifier(n_estimators=100, random_state=42)

    def preparar_dados(self):
        X_norm, y = self.normalizador.processar(feature_range=(0, 1))
        return train_test_split(X_norm, y, test_size=0.2, random_state=42)

    def treinar_e_avaliar(self):
        X_train, X_test, y_train, y_test = self.preparar_dados()
        
        self.modelo.fit(X_train, y_train)
        y_pred = self.modelo.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        classification_rep = classification_report(y_test, y_pred)

        print(f"Accuracy: {accuracy:.2f}")
        print("\nClassification Report:\n", classification_rep)
        
        return self.modelo, y_test, y_pred

if __name__ == "__main__":
    classificador_rf = ClassificadorSinaisVitais()
    modelo_rf, y_test_rf, y_pred_rf = classificador_rf.treinar_e_avaliar()
    
