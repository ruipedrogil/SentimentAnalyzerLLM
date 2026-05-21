"""
Interface Gradio para o Sistema de Análise de Sentimentos Híbrido
Trabalho Final CLN - UBI
"""

import gradio as gr
import json
from sentiment_analyzer import SistemaAnaliseSentimentos

# system init
sistema = SistemaAnaliseSentimentos(modelo_llm="mistral")

# colours for sentiment
COR_LABEL = {
    "Positivo": "#22c55e",
    "Negativo": "#ef4444",
    "Neutro":   "#94a3b8",
}

EMOJI_LABEL = {
    "Positivo": "😊",
    "Negativo": "😞",
    "Neutro":   "😐",
}

# main function
def analisar(texto: str, modelo_llm: str):
    if not texto.strip():
        return (
            "—",
            "—",
            "",
            "",
        )

    resultado = sistema.explicador.modelo != modelo_llm and setattr(
        sistema.explicador, "modelo", modelo_llm
    ) or sistema.analisar(texto)

    # Se setattr retornou None (sempre), re-analisar
    if not isinstance(resultado, dict):
        sistema.explicador.modelo = modelo_llm
        resultado = sistema.analisar(texto)

    label     = resultado["label"]
    confianca = resultado["confianca"]
    scores    = resultado["scores"]
    explicacao = resultado["explicacao"]

    # score bars in html
    barras_html = _render_barras(scores)

    label_formatado = f"{EMOJI_LABEL[label]}  **{label}** — confiança: {confianca:.0%}"

    return label_formatado, barras_html, explicacao


def _render_barras(scores: dict) -> str:
    """Gera HTML com barras de progresso para cada classe."""
    html = "<div style='font-family: monospace; padding: 8px;'>"
    for classe, score in sorted(scores.items(), key=lambda x: -x[1]):
        cor = COR_LABEL.get(classe, "#888")
        pct = score * 100
        html += f"""
        <div style='margin-bottom:10px;'>
            <div style='display:flex; justify-content:space-between; margin-bottom:3px;'>
                <span style='font-weight:600; color:{cor};'>{classe}</span>
                <span>{pct:.1f}%</span>
            </div>
            <div style='background:#e2e8f0; border-radius:6px; height:14px;'>
                <div style='background:{cor}; width:{pct}%; height:14px;
                            border-radius:6px; transition:width 0.4s;'></div>
            </div>
        </div>
        """
    html += "</div>"
    return html


# interface
with gr.Blocks(
    title="Analisador de Sentimentos Híbrido",
    theme=gr.themes.Soft(primary_hue="slate"),
    css="""
    .title-box { text-align:center; padding: 20px 0 10px 0; }
    .title-box h1 { font-size: 1.8rem; font-weight: 700; }
    .title-box p  { color: #64748b; margin-top: 6px; }
    footer { display: none !important; }
    """,
) as demo:

    gr.HTML("""
        <div class='title-box'>
            <h1>🧠 Analisador de Sentimentos Híbrido</h1>
            <p>XLM-RoBERTa (classificação) + LLM local via Ollama (explicação)</p>
            <p style='font-size:0.8rem; color:#94a3b8;'>Trabalho Final CLN · UBI · 2025/26</p>
        </div>
    """)

    with gr.Row():
        with gr.Column(scale=2):
            texto_input = gr.Textbox(
                label="Texto a analisar",
                placeholder="Escreve aqui um comentário, review, frase...",
                lines=5,
            )
            modelo_llm = gr.Dropdown(
                choices=["mistral", "llama3", "llama3.2", "phi3"],
                value="mistral",
                label="Modelo LLM local (Ollama)",
            )
            btn = gr.Button("🔍 Analisar", variant="primary")

        with gr.Column(scale=3):
            label_out  = gr.Markdown(label="Sentimento Detectado")
            barras_out = gr.HTML(label="Distribuição de Scores")
            explicacao_out = gr.Textbox(
                label="💬 Explicação gerada pelo LLM",
                lines=5,
                interactive=False,
            )

    # examples
    gr.Examples(
        examples=[
            ["O produto chegou em perfeito estado e superou todas as minhas expectativas!", "mistral"],
            ["Péssimo serviço, nunca mais volto a esta loja. Fui completamente ignorado.", "mistral"],
            ["A encomenda foi entregue no prazo indicado.", "mistral"],
            ["Adorei a qualidade, mas o preço é um pouco elevado para o que é.", "mistral"],
        ],
        inputs=[texto_input, modelo_llm],
        label="Exemplos",
    )

    btn.click(
        fn=analisar,
        inputs=[texto_input, modelo_llm],
        outputs=[label_out, barras_out, explicacao_out],
    )

if __name__ == "__main__":
    demo.launch(share=False)
