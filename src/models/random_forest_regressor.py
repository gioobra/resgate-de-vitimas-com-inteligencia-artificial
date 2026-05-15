import sys
import pandas as pd
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from resgate.src.normalizacao import NormalizadorSinaisVitais
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error, r2_score

class RegressorSinaisVitais:
    def __init__(self, arquivo_dados='02_treino_sinais_vitais_com_label.txt'):
        self.arquivo_dados = arquivo_dados
        self.normalizador = NormalizadorSinaisVitais(self.arquivo_dados)
        self.modelo = RandomForestRegressor(n_estimators=100, random_state=42)

    def preparar_dados(self):
        X_norm, y = self.normalizador.processar(feature_range=(0, 1))
        return train_test_split(X_norm, y, test_size=0.2, random_state=42)
    
    def treinar_e_avaliar(self):
        X_train, X_test, y_train, y_test = self.preparar_dados()
        
        self.modelo.fit(X_train, y_train)
        y_pred = self.modelo.predict(X_test)

        rmse = root_mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"RMSE: {rmse:.2f}")
        print(f"R-squared Score: {r2:.2f}")

        return self.modelo, y_test, y_pred

if __name__ == "__main__":
    regressor = RegressorSinaisVitais()
    modelo_regressor, y_test, y_pred = regressor.treinar_e_avaliar()
    