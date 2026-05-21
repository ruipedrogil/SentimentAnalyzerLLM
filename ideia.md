## Ideia do Projeto

**Sistema de Análise de Sentimentos em Português com Explicação Generativa, 100% Local**

### O que é

Um sistema que recebe um texto em português (por exemplo, uma review, um comentário, um tweet) e devolve duas coisas:

1. **Uma classificação de sentimento** — Positivo, Negativo ou Neutro — com um score numérico de confiança
2. **Uma explicação em linguagem natural** que justifica essa classificação, identificando as palavras ou expressões que motivaram a decisão

### Como funciona — arquitetura híbrida em dois andares

```
Texto → [Classificador discriminativo] → [LLM generativo] → Resultado
         XLM-RoBERTa                      Mistral 7B
         (rápido, preciso)                (explicação rica)
```

**Andar 1 — Classificador discriminativo (XLM-RoBERTa fine-tuned):**
Um modelo Transformer multilíngue já treinado especificamente para análise de sentimentos. Devolve a classe e um score de confiança em ~100 ms. É a componente que dá rigor numérico ao sistema.

**Andar 2 — LLM generativo local (Mistral 7B via Ollama):**
Recebe o texto original e a classificação do andar 1, e produz uma explicação em português que torna a decisão interpretável para o utilizador. É a componente que dá valor comunicativo ao sistema.

**Interface (Gradio):**
Uma página web local simples onde o utilizador escreve o texto e vê os dois outputs lado a lado.

### Porquê combinar os dois modelos

Esta é a decisão de design central do projeto e responde diretamente ao espírito do enunciado ("sistema que integra linguagem natural, alavancando capacidades operacionais de LLMs"):

- **Classificadores discriminativos** são rápidos e precisos em métricas, mas são caixas-pretas — só dão um rótulo
- **LLMs generativos** são interpretáveis e fluentes, mas mais lentos e menos consistentes em tarefas de classificação
- **A combinação** aproveita o melhor dos dois mundos: precisão quantificável + explicação interpretável

### Como cumpre as restrições do enunciado

| Requisito | Como é cumprido |
|---|---|
| Recursos configurados localmente | Modelos descarregados uma vez e armazenados em disco local (`~/.cache/huggingface/` e `~/.ollama/`) |
| Sem chamadas a APIs em operação | Tudo corre na GPU/CPU local: nem o classificador nem o LLM contactam serviços externos |
| Funciona sem Internet | Após o setup inicial, o sistema é totalmente offline — pode-se desligar a rede e continua a funcionar |
| Integra linguagem natural | Usa dois recursos de NLP complementares (modelo Transformer fine-tuned + LLM generativo) |
| Alavanca capacidades de LLMs | A geração de explicações é feita por um LLM moderno (Mistral 7B) |

### Avaliação prevista

Para defender o trabalho com dados, o sistema é avaliado num dataset rotulado de textos em português, reportando:
- **Accuracy, Precision, Recall, F1** por classe
- **Matriz de confusão**
- **Análise qualitativa** de casos onde a explicação do LLM concorda/discorda do classificador (especialmente útil para discutir sarcasmo, ironia e negações)

### Diferenciador

A maioria dos trabalhos típicos nesta área limita-se a usar um classificador. A combinação de um modelo discriminativo com um LLM generativo local — ambos a correr offline — demonstra:
- Domínio de duas famílias de modelos de NLP (encoder-only e decoder-only)
- Compreensão prática de trade-offs entre precisão e interpretabilidade
- Capacidade de orquestrar múltiplos recursos linguísticos num pipeline coerente

---

Isto serve como base para a Introdução e a secção de Arquitetura do relatório. Quando começares o LaTeX, posso transformar isto em prosa formal com as referências académicas dos modelos (paper do XLM-RoBERTa, do Mistral, etc.).