import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from testes import Testes
from normalizacao import NormalizadorSinaisVitais
from data_loader import DataLoader
#Retirar logica do train split


class ModeloAI:
    def __init__(self, tipo_modelo, metricas):
        self.tipo_modelo = tipo_modelo
        self.metricas_nomes = metricas
        self.modelo = self._configurar_modelo()
        self.loader = DataLoader()
        self.normalizador = NormalizadorSinaisVitais()
        
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def _configurar_modelo(self):
        if self.tipo_modelo == 'Random Forest Classificador':
            return RandomForestClassifier(n_estimators=100, random_state=42)
        elif self.tipo_modelo == 'Random Forest Regressor':
            return RandomForestRegressor(n_estimators=100, random_state=42)
        elif self.tipo_modelo == 'MLP Classificador':
            return MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)
        elif self.tipo_modelo == 'MLP Regressor':
            return MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)

    def carregar_e_preparar_dados(self):
        X, y = self.loader.carregar_dados_treino()
        
        if "MLP" in self.tipo_modelo:
            X = self.normalizador.treinar_scaler(X)
        #ALTERAR PARA TREINAR COM O 02 APENAS    
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    def treinar(self):
        self.modelo(self.X_train, self.y_train)

    def avaliar_validacao(self):
        y_pred = self.modelo.predict(self.X_test)
        return self._calcular_metricas(self.y_test, y_pred)

    def prever_novos_dados(self):
        X_novos = self.loader.carregar_dados_teste()
        
        if "MLP" in self.tipo_modelo:
            X_novos = self.normalizador.aplicar_scaler(X_novos)

        return self.modelo.predict(X_novos)

    def _calcular_metricas(self, y_true, y_pred):
        resultados = {}
        for nome_metrica in self.metricas_nomes:
                #modificar isso porque é confuso
                metrica_func = getattr(testes, nome_metrica)
                resultados[nome_metrica] = metrica_func(y_true, y_pred)
        return resultados
