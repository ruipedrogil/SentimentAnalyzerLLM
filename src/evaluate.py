"""
Avaliação Quantitativa do Sistema de Análise de Sentimentos
Lê o dataset de 300 reviews e avalia o modelo fine-tuned local.
Trabalho Final CLN - UBI
"""

import pandas as pd
from transformers import pipeline
from sklearn.metrics import classification_report, confusion_matrix
import json
import warnings
warnings.filterwarnings("ignore")

def avaliar_csv(caminho_csv="../data/dataset_avaliacao.csv", pasta_modelo="./meu_roberta_b2w"):
    # loading dataset
    
    try:
        df = pd.read_csv(caminho_csv)
    except FileNotFoundError:
        print(f"Ficheiro {caminho_csv} não encontrado.")
        return

   # loading our fine-tined model locally
    pipe = pipeline(
        "text-classification", 
        model=pasta_modelo, 
        tokenizer=pasta_modelo, 
        truncation=True, 
        max_length=128,
        device=0 # Usa a GPU se estiver disponível
    )

    y_true = []
    y_pred = []
    resultados_detalhados = []

    # Map (apanha labels do hugging face ou LABEL_X)
    mapa_labels = {
        "LABEL_0": "Negativo", "LABEL_1": "Neutro", "LABEL_2": "Positivo",
        "negative": "Negativo", "neutral": "Neutro", "positive": "Positivo",
        "Negative": "Negativo", "Neutral": "Neutro", "Positive": "Positivo"
    }

    
    print(f"Avaliação do modelo fine-tuned RoBERTa — {len(df)} exemplos")
    for index, row in df.iterrows():
        texto = row['texto']
        # Força Capitalização
        label_esperado = str(row['label']).capitalize()

        # Inferência
        resultado = pipe(texto)[0]
        
        # Mapeia e Força Capitalização
        label_pred = mapa_labels.get(resultado['label'], resultado['label']).capitalize()

        y_true.append(label_esperado)
        y_pred.append(label_pred)

        correto = label_pred == label_esperado
        
        resultados_detalhados.append({
            "texto": texto,
            "esperado": label_esperado,
            "predito": label_pred,
            "confianca": resultado["score"],
            "correto": correto,
        })

    # metrics 
    labels_ordem = ["Negativo", "Neutro", "Positivo"]
    
    print("Relatório de classificação final (Scikit-Learn)")
    print(classification_report(y_true, y_pred, labels=labels_ordem))

    print("\nMatriz de Confusão:")
    cm = confusion_matrix(y_true, y_pred, labels=labels_ordem)
    
    print(f"{'':12}", end="")
    for l in labels_ordem:
        print(f"{l:12}", end="")
    print()
    for i, row in enumerate(cm):
        print(f"{labels_ordem[i]:12}", end="")
        for val in row:
            print(f"{val:<12}", end="")
        print()

    # save results
    output = {
        "total": len(df),
        "corretos": sum(1 for r in resultados_detalhados if r["correto"]),
        "accuracy": sum(1 for r in resultados_detalhados if r["correto"]) / len(df),
        "detalhes": resultados_detalhados,
    }

    with open("resultados_avaliacao.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nAccuracy geral do nosso modelo: {output['accuracy']:.1%} ({output['corretos']}/{output['total']})")
    print(f"A título de comparação, o modelo original tinha 64.3%.")

if __name__ == "__main__":
    avaliar_csv()