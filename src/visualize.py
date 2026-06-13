import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# config
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({'font.size': 11, 'figure.titlesize': 14})


# gráfico evolução da accuracy
fig1, ax1 = plt.subplots(figsize=(6, 4.5), dpi=300)
modelos = ['Modelo Base\n(Twitter-XLM-RoBERTa)', 'Sistema Proposto\n(Após Fine-Tuning Local)']
exatidao = [64.3, 78.0]

bars = ax1.bar(modelos, exatidao, color=['#64748b', '#22c55e'], width=0.5, edgecolor='#334155', linewidth=1.2)

# adicionar os valores no topo das barras
for bar in bars:
    height = bar.get_height()
    ax1.annotate(f'{height}%',
                 xy=(bar.get_x() + bar.get_width() / 2, height),
                 xytext=(0, 3),  # 3 points vertical offset
                 textcoords="offset points",
                 ha='center', va='bottom', fontweight='bold')

ax1.set_ylabel('Acurácia Geral (Accuracy %)', fontweight='bold')
ax1.set_ylim(0, 100)
ax1.set_title('Impacto da Adaptação de Domínio no Dataset de Teste', pad=15, fontweight='bold')
plt.tight_layout()
plt.savefig('evolucao_exatidao.png', bbox_inches='tight')
plt.close()


# gráfico métricas detalhadas por classes
fig2, ax2 = plt.subplots(figsize=(8, 5), dpi=300)

classes = ['Negativo', 'Neutro', 'Positivo']
precision = [0.79, 0.72, 0.82]
recall = [0.88, 0.63, 0.83]
f1_score = [0.83, 0.67, 0.83]

x = np.arange(len(classes))
width = 0.25

rects1 = ax2.bar(x - width, precision, width, label='Precision', color='#0284c7', edgecolor='#0369a1')
rects2 = ax2.bar(x, recall, width, label='Recall', color='#22c55e', edgecolor='#15803d')
rects3 = ax2.bar(x + width, f1_score, width, label='F1-Score', color='#3b82f6', edgecolor='#1d4ed8')

# adicionar labels e estilização
ax2.set_ylabel('Pontuação (0 - 1.0)', fontweight='bold')
ax2.set_title('Métricas de Performance por Classe (XLM-RoBERTa Fine-tuned)', pad=15, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(classes, fontweight='bold')
ax2.set_ylim(0, 1.1)
ax2.legend(frameon=True, facecolor='white', edgecolor='#e2e8f0')

# função para colocar os valores no topo das barras pequenas
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax2.annotate(f'{height:.2f}',
                     xy=(rect.get_x() + rect.get_width() / 2, height),
                     xytext=(0, 2),
                     textcoords="offset points",
                     ha='center', va='bottom', fontsize=9)

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

plt.tight_layout()
plt.savefig('metricas_por_classe.png', bbox_inches='tight')
plt.close()

print("Gráficos gerados com sucesso.")