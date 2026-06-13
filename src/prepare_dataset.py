import pandas as pd


# usa-se diretamente o caminho do GitHub que funciona
url_github = "https://raw.githubusercontent.com/americanas-tech/b2w-reviews01/main/B2W-Reviews01.csv"

# ler o CSV diretamente (low_memory=False para evitar warnings)
df = pd.read_csv(url_github, low_memory=False)

# Manter apenas colunas relevantes e remover linhas vazias (nulls)
df = df[['review_text', 'overall_rating']].dropna()
df = df[df['review_text'].str.len() > 10]  # filtrar reviews muito curtas

# Mapear rating (1-5) → label de sentimento
def rating_to_label(r):
    if r <= 2:
        return 'negativo'
    elif r == 3:
        return 'neutro'
    else:
        return 'positivo'

df['label'] = df['overall_rating'].apply(rating_to_label)
df = df.rename(columns={'review_text': 'texto'})[['texto', 'label']]

# amostra balanced, 100 por classe (300 exemplos no total)
N_POR_CLASSE = 100

# usar .sample() diretamente no groupby (evita o bug do .apply() na versão do pandas)
sample = df.groupby('label').sample(n=N_POR_CLASSE, random_state=42)

# baralhar a ordem aleatoriamente e dar reset ao index de forma segura
sample = sample.sample(frac=1, random_state=42).reset_index(drop=True)

# guardar
sample.to_csv('dataset_avaliacao.csv', index=False, encoding='utf-8')

print(f"\n[OK] Dataset criado com sucesso: {len(sample)} exemplos")
print("\nDistribuição por classe:")
print(sample['label'].value_counts())
print(f"\nGuardado em: dataset_avaliacao.csv")
print("\nPrimeiros exemplos:")
print(sample.head(3).to_string(index=False))