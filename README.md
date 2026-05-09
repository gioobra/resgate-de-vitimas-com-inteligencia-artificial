pyproject.toml -> requirements.txt
dicas MLP
    *Pesos* **Treinamento por batelada ou por padrão**
        Inicializar aleatoriamente os pesos no invervalo [-1, 1]
    *Bias* **Lembrar de incluir os limiares no conjunto de pesos** 
        associados com entradas fixa em -1
    *Entradas* **Normalizar as entradas:**
        [0.1, 0.9] sigmoide
        [-0.9, 0.9] tangente hiperbólica
    *Tx Aprend.* **Utilizar valores pequenos para taxa de aprendizado**
        n E [0.01, 0.1]
    **Redução de Overfitting**
        Dropout
        Regularização L1 e L2
        *Validação cruzada*
    **Normalização das features**
        Normalizar no pré-processamento
        Escala Min-Max
            Sigmoide Xmin = 0,Xmax = 1
            TanH Xmin = -1,Xmax = 1
            Xnorm = (X - Xmin)/(Xmax - Xmin)