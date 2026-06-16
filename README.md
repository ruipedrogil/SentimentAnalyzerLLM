# 🧠 Sistema Híbrido de Análise de Sentimentos em Português com Modelos Locais

[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org)
[![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-orange)](https://ollama.com)
[![Framework](https://img.shields.io/badge/UI-Gradio-FF5722?logo=gradio&logoColor=white)](https://gradio.app)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

Este repositório contém o código-fonte e a documentação do projeto desenvolvido para a unidade curricular de **Computação em Linguagem Natural** do Mestrado em Engenharia Informática na **Universidade da Beira Interior (UBI)**.

O sistema implementa uma arquitetura modular baseada num *pipeline* sequencial de dois níveis independentes, projetada para operar de forma **100% local e offline**, garantindo a total privacidade dos dados do utilizador.

---

## 🗺️ Visão Geral da Arquitetura

O ecossistema divide-se em duas camadas principais:
1. **Nível 1 (Classificador Discriminativo):** O modelo *encoder-only* **XLM-RoBERTa** (fine-tuned localmente) processa a string de entrada quase instantaneamente, extraindo a predição categórica (Positivo, Negativo ou Neutro) e a sua respetiva distribuição probabilística de confiança.
2. **Nível 2 (Explicação Generativa Local):** O módulo *Prompt Builder* encapsula os dados numéricos do Nível 1 e reconstrói um prompt estruturado[cite: 145, 191]. [cite_start]Este é enviado ao LLM **Mistral 7B** (executado via *Ollama*), que formula uma justificação textual em português europeu, identificando as palavras-chave determinantes da classificação (IA Explicável - XAI).

---

## 📊 Resultados Experimentais

O classificador foi treinado localmente com 3000 exemplos do *corpus* público **B2W-Reviews** (avaliações reais de e-commerce em português) e avaliado num conjunto de teste isolado e balanceado de 300 amostras.

### Evolução de Desempenho (Adaptação de Domínio)
* **Modelo Base (`cardiffnlp/twitter-xlm-roberta-base-sentiment`):** 64.3% de Acurácia.
* **Sistema Proposto (Após Fine-Tuning Local):** **78.0% de Acurácia**.

### Métricas Finais Detalhadas por Classe

| Classe | Precision | Recall | F1-Score | Suporte |
| :--- | :---: | :---: | :---: | :---: |
| **Negativo** | 0.79 | 0.88 | 0.83 | 100 |
| **Neutro** | 0.72 | 0.63 | 0.67 | 100 |
| **Positivo** | 0.82 | 0.83 | 0.83 | 100 |
| **Acurácia Geral (Accuracy)** | | | **78.0%** | **300** |

*Nota: A classe Neutra constitui o maior desafio analítico devido à prevalência de sentimentos mistos na mesma frase (e.g., elogio ao produto, mas crítica severa à logística de entrega). É precisamente nesta fronteira semântica que a componente explicativa do Mistral 7B mitiga a opacidade de "caixa-preta", contextualizando o veredicto*

---

## 📂 Estrutura de Ficheiros do Projeto

O código-fonte está modularizado de forma a separar claramente as responsabilidades de engenharia de dados, treino, inferência e interface:

```text
projeto/
├── app.py                  # Interface web interativa construída em Gradio
├── sentiment_analyzer.py   # Orquestrador do pipeline híbrido (RoBERTa + Mistral)
├── prepare_dataset.py      # Extração, pré-processamento e split do corpus B2W
├── train_model.py          # Script de fine-tuning local recorrendo à Hugging Face
├── evaluate.py             # Script para avaliação quantitativa e geração de métricas
├── visualize.py            # Geração dos gráficos de performance presentes no relatório
└── dataset_avaliacao.csv   # Conjunto de teste isolado (300 amostras estáveis)
```

---

## 🛠️ Hiperparâmetros de Treino

O processo de otimização local foi executado em ambiente PyTorch com as seguintes configurações:

- Taxa de Aprendizagem: $2 \times 10^{-5}$;
- Dimensão do Lote (Batch Size): 8 (Treino e Avaliação);
- Otimizador: AdamW;
- Decaimento de Pesos (Weight Decay): 0.01;
- Prevenção de Overfitting: Early Stopping com paciência de 2 épocas;
- Otimização de VRAM: Alocação dinâmica de matrizes via DataCollatorWithPadding.

---

## 🚀 Como Executar Localmente

#### Pré-requisitos:

- Certifica-te de ter o Python 3.10 ou superior instalado.
- Instala e inicializa o Ollama no teu sistema.
- Descarrega o modelo Mistral executando o seguinte comando no terminal:

```
ollama run mistral
```

#### Instalação:

- Clona este repositório:

  ```bash
   git clone [https://github.com/ruipedrogil/SentimentAnalyzerLLM.git](https://github.com/ruipedrogil/SentimentAnalyzerLLM.git)
   cd SentimentAnalyzerLLM
  ```

- Instala as dependências necessárias:

  ```
  pip install -r requirements.txt
  ```

#### Execução do Sistema:

- Para lançar a interface gráfica interativa do Gradio no teu browser local (http://127.0.0.1:7860), executa:

  ```
  python app.py
  ```
