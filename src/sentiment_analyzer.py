"""
Sistema de Análise de Sentimentos Híbrido
Combina XLM-RoBERTa (classificação) + Ollama/Mistral (explicação)
Trabalho Final CLN - UBI
"""

from transformers import pipeline
import ollama
import json


# classifier (HuggingFace - XLM-RoBERTa)
class ClassificadorSentimentos:
    """
    Usa o modelo cardiffnlp/twitter-xlm-roberta-base-sentiment
    Treinado para Positivo / Negativo / Neutro em múltiplas línguas.
    """

    LABEL_MAP = {
        "positive": "Positivo",
        "negative": "Negativo",
        "neutral":  "Neutro",
    }

    def __init__(self):
        # loading XLM-RoBERTa model
        self.pipe = pipeline(
            task="text-classification",
            model="./meu_roberta_b2w",
            top_k=None,          # devolve scores para todas as classes
            truncation=True,
            max_length=512,
        )
        print("Modelo carregado com sucesso.")

    def classificar(self, texto: str) -> dict:
        """
        Retorna:
          {
            "label": "Positivo" | "Negativo" | "Neutro",
            "confianca": 0.92,        # score da classe vencedora
            "scores": {               # scores de todas as classes
                "Positivo": 0.92,
                "Negativo": 0.05,
                "Neutro":   0.03,
            }
          }
        """
        resultados = self.pipe(texto)[0]   # lista de {label, score}
        scores = {
            self.LABEL_MAP.get(r["label"].lower(), r["label"]): round(r["score"], 4)
            for r in resultados
        }
        label_top = max(scores, key=scores.get)
        return {
            "label": label_top,
            "confianca": scores[label_top],
            "scores": scores,
        }


# explainer llm (Ollama - Mistral / LLaMA3)
class ExplicadorLLM:
    """
    Usa um LLM local via Ollama para gerar uma explicação
    em linguagem natural para a classificação obtida.
    """

    PROMPT_TEMPLATE = """\
Foste dado o seguinte texto para análise de sentimentos:

TEXTO: "{texto}"

Um classificador automático determinou que o sentimento é: {label} (confiança: {confianca:.0%}).

Com base no texto acima, explica em 2 a 3 frases, em português europeu:
1. Por que razão o sentimento foi classificado como {label}.
2. Quais as palavras ou expressões chave que mais contribuíram para esta classificação.

Responde de forma clara e direta, sem repetir o texto na íntegra.
"""

    def __init__(self, modelo: str = "mistral"):
        self.modelo = modelo
        print(f"Explicador configurado com modelo '{modelo}' via Ollama.")

    def explicar(self, texto: str, classificacao: dict) -> str:
        """
        Gera uma explicação em linguagem natural para a classificação.
        """
        prompt = self.PROMPT_TEMPLATE.format(
            texto=texto,
            label=classificacao["label"],
            confianca=classificacao["confianca"],
        )
        try:
            resposta = ollama.chat(
                model=self.modelo,
                messages=[{"role": "user", "content": prompt}],
            )
            return resposta["message"]["content"].strip()
        except Exception as e:
            return f"[Ollama indisponível] Erro: {e}"



# combined hybrid system
class SistemaAnaliseSentimentos:
    """
    Pipeline completo:
      texto → XLM-RoBERTa (classificação + confiança)
            → Ollama/Mistral (explicação em linguagem natural)
    """

    def __init__(self, modelo_llm: str = "mistral"):
        self.classificador = ClassificadorSentimentos()
        self.explicador    = ExplicadorLLM(modelo=modelo_llm)

    def analisar(self, texto: str) -> dict:
        """
        Retorna um dicionário com todos os resultados.
        """
        if not texto or not texto.strip():
            return {"erro": "Texto vazio."}

        # step 1: classify
        classificacao = self.classificador.classificar(texto)

        # step 2: explain
        explicacao = self.explicador.explicar(texto, classificacao)

        return {
            "texto":      texto,
            "label":      classificacao["label"],
            "confianca":  classificacao["confianca"],
            "scores":     classificacao["scores"],
            "explicacao": explicacao,
        }

    def analisar_batch(self, textos: list[str]) -> list[dict]:
        """Analisa uma lista de textos."""
        return [self.analisar(t) for t in textos]



# terminal demo wiithout (GUI which is in app.py)
if __name__ == "__main__":
    sistema = SistemaAnaliseSentimentos(modelo_llm="mistral")

    exemplos = [
        "O produto chegou em perfeito estado e superou todas as minhas expectativas!",
        "Péssimo serviço, nunca mais volto a esta loja.",
        "O pacote chegou dentro do prazo previsto.",
    ]

    print("\n" + "="*60)
    print("SISTEMA DE ANÁLISE DE SENTIMENTOS HÍBRIDO")
    print("="*60)

    for texto in exemplos:
        resultado = sistema.analisar(texto)
        print(f"\nTexto: {resultado['texto']}")
        print(f"Sentimento: {resultado['label']} ({resultado['confianca']:.0%})")
        print(f"Scores: {json.dumps(resultado['scores'], ensure_ascii=False)}")
        print(f"Explicação: {resultado['explicacao']}")
        print("-"*60)
