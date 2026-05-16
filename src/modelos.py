from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
import sys
from pathlib import Path
import sys
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor

sys.path.append(str(Path(__file__).parent))
from testes import Testes
from normalizacao import NormalizadorSinaisVitais
from data_loader import DataLoader


class ModeloAI:
    def __init__(self, tipo_modelo, metricas, config):
        self.tipo_modelo = tipo_modelo
        self.metricas_nomes = metricas
        self.config = config
        self.modelo = self._configurar_modelo()
        self.loader = DataLoader()
        self.normalizador = NormalizadorSinaisVitais()
        self.testes = Testes()

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def _configurar_modelo(self):
        rf = self.config.get('random_forest', {})
        mlp = self.config.get('mlp', {})

        if self.tipo_modelo == 'Random Forest Classificador':
            return RandomForestClassifier(
                n_estimators=rf.get('n_estimators', 100),
                max_depth=rf.get('max_depth', None),
                criterion=rf.get('criterion', 'gini'),
                random_state=rf.get('random_state', 42)
            )
        elif self.tipo_modelo == 'Random Forest Regressor':
            return RandomForestRegressor(
                n_estimators=rf.get('n_estimators', 100),
                max_depth=rf.get('max_depth', None),
                random_state=rf.get('random_state', 42)
            )
        elif self.tipo_modelo == 'MLP Classificador':
            return MLPClassifier(
                hidden_layer_sizes=tuple(mlp.get('hidden_layer_sizes', [16, 8])),
                activation=mlp.get('activation', 'relu'),
                solver=mlp.get('solver', 'adam'),
                learning_rate_init=mlp.get('learning_rate_init', 0.001),
                max_iter=mlp.get('max_iter', 1000),
                random_state=mlp.get('random_state', 42)
            )
        elif self.tipo_modelo == 'MLP Regressor':
            return MLPRegressor(
                hidden_layer_sizes=tuple(mlp.get('hidden_layer_sizes', [16, 8])),
                activation=mlp.get('activation', 'relu'),
                solver=mlp.get('solver', 'adam'),
                learning_rate_init=mlp.get('learning_rate_init', 0.001),
                max_iter=mlp.get('max_iter', 1000),
                random_state=mlp.get('random_state', 42)
            )

    def carregar_e_preparar_dados(self):
        split_cfg = self.config.get('split', {})
        X, y = self.loader.carregar_dados_treino()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=split_cfg.get('test_size', 0.2),
            random_state=split_cfg.get('random_state', 42)
        )

        if "MLP" in self.tipo_modelo:
            X_train = self.normalizador.normalizar_treino(X_train)
            X_test = self.normalizador.normalizar_teste(X_test)

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def treinar(self):
        # .fit() é o método padrão do sklearn para treinar qualquer modelo.
        self.modelo.fit(self.X_train, self.y_train)

    def avaliar_validacao(self):
        y_pred = self.modelo.predict(self.X_test)
        return self._calcular_metricas(self.y_test, y_pred)

    def prever_novos_dados(self):
        X_novos = self.loader.carregar_dados_teste()

        if "MLP" in self.tipo_modelo:
            X_novos = self.normalizador.normalizar_teste(X_novos)

        return self.modelo.predict(X_novos)

    def _calcular_metricas(self, y_true, y_pred):
        resultados = {}
        for nome_metrica in self.metricas_nomes:
            metrica_func = getattr(self.testes, nome_metrica)
            resultados[nome_metrica] = metrica_func(y_true, y_pred)
        return resultados