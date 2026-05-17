class NormalizadorSinaisVitais:
    def __init__(self):
        self.min_ = None
        self.max_ = None

    def normalizar_treino(self, X):
        self.min_ = X.min()
        self.max_ = X.max()
        return self._aplicar(X)

    def normalizar_teste(self, X):
        if self.min_ is None or self.max_ is None:
            raise ValueError("Chame normalizar_treino() antes de normalizar_teste().")
        return self._aplicar(X)

    def _aplicar(self, X):
        denominador = self.max_ - self.min_
        denominador = denominador.replace(0, 1)  # evita divisão por zero
        return (X - self.min_) / denominador