"""
Treino (Fine-Tuning) do XLM-RoBERTa para Análise de Sentimentos em E-commerce
Trabalho Final CLN - UBI
"""

import os
import argparse
import pandas as pd
import numpy as np
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer, 
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    pipeline
)

def treinar(args):
    # prepare data train
    url_github = "https://raw.githubusercontent.com/americanas-tech/b2w-reviews01/main/B2W-Reviews01.csv"
    df_all = pd.read_csv(url_github, low_memory=False)[['review_text', 'overall_rating']].dropna()
    df_all = df_all[df_all['review_text'].str.len() > 10]
    df_all = df_all.rename(columns={'review_text': 'texto'})

    # load the test dataset to remove it from the train one (avoid overfitting)
    if os.path.exists(args.test_data_path):
        df_teste = pd.read_csv(args.test_data_path)
        tamanho_antes = len(df_all)
        df_all = df_all[~df_all['texto'].isin(df_teste['texto'])]
        print(f"Proteção ativa: {tamanho_antes - len(df_all)} exemplos de teste removidos do treino.")
    else:
        print(f" Dataset de teste ({args.test_data_path}) não encontrado.")

    # Mapeamento do modelo original: 0=Negative, 1=Neutral, 2=Positive
    def rating_to_label_id(r):
        if r <= 2: return 0
        elif r == 3: return 1
        else: return 2

    df_all['label'] = df_all['overall_rating'].apply(rating_to_label_id)

    # Dataset de Treino balanceado (N exemplos exatos por classe = sem necessidade de Class Weights!)
    df_treino = df_all.groupby('label').sample(n=args.n_treino, random_state=42).reset_index(drop=True)

    # Validação (para o modelo testar no final de cada época)
    df_resto = df_all[~df_all['texto'].isin(df_treino['texto'])]
    df_val = df_resto.groupby('label').sample(n=150, random_state=42).reset_index(drop=True)

    dataset_treino = Dataset.from_pandas(df_treino[['texto', 'label']])
    dataset_val = Dataset.from_pandas(df_val[['texto', 'label']])

    #  load model and tokenizer
    modelo_base = "cardiffnlp/twitter-xlm-roberta-base-sentiment"

    tokenizer = AutoTokenizer.from_pretrained(modelo_base)
    modelo = AutoModelForSequenceClassification.from_pretrained(modelo_base, num_labels=3)

    def preprocess_function(examples):
        return tokenizer(examples["texto"], truncation=True, max_length=128)

    tokenized_treino = dataset_treino.map(preprocess_function, batched=True)
    tokenized_val = dataset_val.map(preprocess_function, batched=True)

    # define metrics for eval
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        acc = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average="macro")
        return {"accuracy": acc, "f1_macro": f1}

    # early stopping and config for fine-tune

    training_args = TrainingArguments(
        output_dir=args.model_out,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,         # keeps the best model
        metric_for_best_model="accuracy",    # decide which is better based on accuracy
        save_total_limit=2,                  # don't fill the disc with all weights for all epochs
    )

    data_collator = DataCollatorWithPadding(tokenizer)

    trainer = Trainer(
        model=modelo,
        args=training_args,
        train_dataset=tokenized_treino,
        eval_dataset=tokenized_val,
        data_collator=data_collator,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)] # stop if doesent improve in the next 2 epochs
    )

    # start train
    trainer.train()

    print(f"\nTreino concluído, a guardar modelo final em: {args.model_out}")
    trainer.save_model(args.model_out)
    tokenizer.save_pretrained(args.model_out)
    
    return args.model_out

def avaliar_modelo_finetuned(model_path, test_data_path):
    """Lê o modelo treinado e avalia-o no teu dataset de avaliação de 300 linhas."""
    if not os.path.exists(test_data_path):
        print(f"\n[ERRO] Dataset de teste não encontrado para avaliação final: {test_data_path}")
        return

    print(f"\n{'='*60}")
    print(f"  AVALIAÇÃO FINAL: Modelo Fine-Tuned vs Exame Final (300 reviews)")
    print(f"{'='*60}")

    # Desativa o aviso chato da pipeline sequencial
    import warnings
    warnings.filterwarnings("ignore")

    pipe = pipeline(
        "text-classification", 
        model=model_path, 
        tokenizer=model_path, 
        truncation=True, 
        max_length=128,
        device=0 # Usa GPU se disponível
    )

    df_teste = pd.read_csv(test_data_path)
    
    y_true, y_pred = [], []
    mapa_labels = {
        "LABEL_0": "Negativo", "LABEL_1": "Neutro", "LABEL_2": "Positivo",
        "negative": "Negativo", "neutral": "Neutro", "positive": "Positivo"
    }

    for _, row in df_teste.iterrows():
        texto = row['texto']
        label_esperado = str(row['label']).capitalize()
        
        resultado = pipe(texto)[0]
        pred_label = mapa_labels.get(resultado['label'], resultado['label']).capitalize()
        
        y_true.append(label_esperado)
        y_pred.append(pred_label)

    acc = accuracy_score(y_true, y_pred)
    
    print(f"\n  Accuracy Final: {acc:.1%}  (Lembrança: O base tinha 64.3%!)")
    print("\n" + classification_report(y_true, y_pred, labels=["Negativo", "Neutro", "Positivo"]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=4, help="Número de passagens pelos dados")
    parser.add_argument("--batch_size", type=int, default=8, help="Tamanho do batch")
    parser.add_argument("--lr", type=float, default=2e-5, help="Taxa de aprendizagem")
    parser.add_argument("--n_treino", type=int, default=1000, help="Reviews por classe (x3 = total)") # dar-lhe mais exemplos de treino, Se usarmos --n_treino 1000 o script vai procurar exatamente 1000 reviews Positivas, vai procurar exatamente 1000 reviews Neutras e vai procurar exatamente 1000 reviews Negativas fazendo o Total de treino = 3000 frases.
    parser.add_argument("--model_out", type=str, default="./meu_roberta_b2w", help="Pasta para guardar")
    parser.add_argument("--test_data_path", type=str, default="../data/dataset_avaliacao.csv", help="Caminho do teu CSV")
    parser.add_argument("--so_avaliar", action="store_true", help="Se quiseres apenas testar o modelo já treinado")
    args = parser.parse_args()

    if not args.so_avaliar:
        pasta_modelo = treinar(args)
    else:
        pasta_modelo = args.model_out
        
    avaliar_modelo_finetuned(pasta_modelo, args.test_data_path)