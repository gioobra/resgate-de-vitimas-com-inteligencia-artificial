import yaml
import pandas as pd
from modelos import ModeloAI

def carregar_config(caminho="config.yaml"):
    with open(caminho, "r") as f:
        return yaml.safe_load(f)

def imprimir_metricas(resultados):
    print("\n--- Resultados ---")
    for nome, valor in resultados.items():
        if nome == "confusion_matrix":
            print(f"\nMatriz de Confusão (linhas=real, colunas=predito, classes 1-4):")
            print(valor)
        else:
            print(f"  {nome}: {valor:.4f}")

def salvar_predicoes(predicoes, caminho_saida):
    ids = range(1, len(predicoes) + 1)
    df = pd.DataFrame({"i": ids, "predicao": predicoes})
    df.to_csv(caminho_saida, index=False)
    print(f"\nPredicoes salvas em: {caminho_saida}")

def main():
    config = carregar_config()

    tipo_modelo = config["modelo_ativo"]
    eh_classificador = "Classificador" in tipo_modelo
    metricas = (
        config["metricas"]["classificacao"]
        if eh_classificador
        else config["metricas"]["regressao"]
    )

    print(f"Modelo: {tipo_modelo}")
    print(f"Métricas: {metricas}")

    modelo = ModeloAI(tipo_modelo, metricas, config)

    print("\n[1/4] Carregando e preparando dados...")
    modelo.carregar_e_preparar_dados()

    print("[2/4] Treinando...")
    modelo.treinar()

    print("[3/4] Avaliando no conjunto de teste...")
    resultados = modelo.avaliar_validacao()
    imprimir_metricas(resultados)

    print("\n[4/4] Prevendo dados de teste cego...")
    predicoes = modelo.prever_novos_dados()
    salvar_predicoes(predicoes, config["dados"]["saida"])

if __name__ == "__main__":
    main()