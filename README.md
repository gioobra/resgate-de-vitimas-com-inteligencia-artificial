
# Dicas
# 🫀 Classificação de Sinais Vitais com ML

Sistema de aprendizado de máquina para classificação e regressão de sinais vitais, com suporte a **Random Forest** e **MLP (Multi-Layer Perceptron)**, configurável via arquivo YAML.

---

## 📁 Estrutura do Projeto

```
.
├── config.yaml                          # Configurações do modelo e dados
├── main.py                              # Ponto de entrada da aplicação
├── data/
│   ├── 01_treino_sinais_vitais_sem_label.txt   # Dados de teste cego (sem rótulo)
│   └── 02_treino_sinais_vitais_com_label.txt   # Dados de treino (com rótulo)
└── src/
    ├── data_loader.py                   # Carregamento dos dados
    ├── modelos.py                       # Orquestração do pipeline de ML
    ├── normalizacao.py                  # Normalização Min-Max
    └── testes.py                        # Cálculo de métricas
```

---

## ⚙️ Configuração (`config.yaml`)

O comportamento do sistema é controlado pelo arquivo `config.yaml`. Exemplo de estrutura esperada:

```yaml
modelo_ativo: "Random Forest Classificador"  # ou: Random Forest Regressor, MLP Classificador, MLP Regressor

dados:
  saida: "predicoes.csv"

metricas:
  classificacao: [accuracy, precision, recall, f1, confusion_matrix]
  regressao: [rmse, mae, r2]

split:
  test_size: 0.2
  random_state: 42

random_forest:
  n_estimators: 100
  max_depth: null
  criterion: "gini"
  random_state: 42

mlp:
  hidden_layer_sizes: [16, 8]
  activation: "relu"
  solver: "adam"
  learning_rate_init: 0.001
  max_iter: 1000
  random_state: 42
```

---

## 🚀 Como Executar

### 1. Instalar dependências

```bash
pip install scikit-learn pandas numpy pyyaml
```

### 2. Organizar os dados

Coloque os arquivos `.txt` no diretório `data/`:
- `02_treino_sinais_vitais_com_label.txt` — dados de treino com rótulos na última coluna
- `01_treino_sinais_vitais_sem_label.txt` — dados de teste sem rótulo (para predição final)

> Os arquivos devem ser CSVs sem cabeçalho. A primeira coluna é ignorada (ID), e a última coluna é o rótulo (apenas no arquivo de treino).

### 3. Configurar o modelo

Edite o `config.yaml` escolhendo o `modelo_ativo` e ajustando os hiperparâmetros desejados.

### 4. Executar

```bash
python main.py
```

---

## 🔄 Pipeline de Execução

```
1. Carrega config.yaml
2. Carrega dados de treino e divide em treino/validação
3. (MLP apenas) Normaliza os dados com Min-Max
4. Treina o modelo escolhido
5. Avalia no conjunto de validação → imprime métricas
6. Prevê dados de teste cego → salva CSV com predições
```

---

## 📊 Modelos Disponíveis

| `modelo_ativo`                | Tipo           | Normalização |
|-------------------------------|----------------|--------------|
| `Random Forest Classificador` | Classificação  | Não          |
| `Random Forest Regressor`     | Regressão      | Não          |
| `MLP Classificador`           | Classificação  | Sim (Min-Max)|
| `MLP Regressor`               | Regressão      | Sim (Min-Max)|

---

## 📈 Métricas Suportadas

**Classificação**
- `accuracy` — Acurácia geral
- `precision` — Precisão (macro)
- `recall` — Revocação (macro)
- `f1` — F1-Score (macro)
- `confusion_matrix` — Matriz de confusão (classes 1–4)

**Regressão**
- `rmse` — Raiz do Erro Quadrático Médio
- `mae` — Erro Absoluto Médio
- `r2` — Coeficiente de Determinação

---

## 📤 Saída

Um arquivo CSV é gerado no caminho definido em `config.yaml` (`dados.saida`), com o seguinte formato:

```csv
i,predicao
1,3
2,1
3,4
...
```

---

## 🧩 Arquitetura dos Módulos

| Módulo             | Responsabilidade                                                  |
|--------------------|-------------------------------------------------------------------|
| `main.py`          | Orquestra o fluxo completo e exibe resultados                    |
| `modelos.py`       | Classe `ModeloAI`: configura, treina, avalia e prevê             |
| `data_loader.py`   | Classe `DataLoader`: lê os arquivos `.txt` do diretório `data/`  |
| `normalizacao.py`  | Classe `NormalizadorSinaisVitais`: normalização Min-Max stateful |
| `testes.py`        | Classe `Testes`: wrappers estáticos das métricas do scikit-learn |
