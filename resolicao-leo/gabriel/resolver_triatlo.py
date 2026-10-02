from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


CSV_PATH = Path(__file__).with_name("triatlo_bruto.csv")
RANDOM_STATE = 42
TEST_SIZE = 0.30
N_NEIGHBORS = 5
N_CLUSTERS = 2

dados = pd.read_csv("triatlo_bruto.csv")
if "concluiu_prova" not in dados.columns:
    raise ValueError("A coluna 'concluiu_prova' nao foi encontrada.")

print("Dimensoes originais (Linhas x Colunas):", dados.shape)

for coluna in ["sexo", "distancia_prova", "tipo_natacao"]:
    dados[coluna] = dados[coluna].astype("string").str.strip().str.title()
    dados[coluna] = dados[coluna].fillna(dados[coluna].mode()[0])

for coluna in [
    "idade",
    "altura_cm",
    "peso_kg",
    "anos_experiencia",
    "horas_natacao_semana",
    "horas_ciclismo_semana",
    "horas_corrida_semana",
    "vo2_max",
    "fc_repouso",
    "fc_maxima",
    "tss_semanal_medio",
    "temperatura_prova_c",
    "altimetria_corrida_m",
    "altimetria_ciclismo_m",
]:
    dados[coluna] = dados[coluna].fillna(dados[coluna].median())

alvo = dados["concluiu_prova"]
atributos = dados.drop(columns=["atleta_id", "concluiu_prova"])
atributos = pd.get_dummies(
    atributos,
    columns=["sexo", "distancia_prova", "tipo_natacao"],
    drop_first=True,
    dtype=int,
)

treino, teste = train_test_split(
    atributos.index,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=alvo,
)

escala = StandardScaler()
atributos_treino = escala.fit_transform(atributos.loc[treino])
atributos_teste = escala.transform(atributos.loc[teste])

modelo_knn = KNeighborsClassifier(n_neighbors=N_NEIGHBORS)
modelo_knn.fit(atributos_treino, alvo.loc[treino])
previsoes = modelo_knn.predict(atributos_teste)
acuracia = accuracy_score(alvo.loc[teste], previsoes)
relatorio = classification_report(alvo.loc[teste], previsoes)

modelo_kmeans = KMeans(
    n_clusters=N_CLUSTERS,
    random_state=RANDOM_STATE,
    n_init=10,
)
grupos = modelo_kmeans.fit_predict(atributos_treino)
silhueta = silhouette_score(atributos_treino, grupos)

print(f"Arquivo: {CSV_PATH.name}")
print(f"Dimensoes: {dados.shape[0]} linhas x {dados.shape[1]} colunas")
print(f"K-NN - acuracia no teste: {acuracia * 100:.2f}%")
print("Relatorio de classificacao:")
print(relatorio)
print(f"K-Means - k escolhido: {N_CLUSTERS}")
print(f"K-Means - silhueta: {silhueta:.4f}")
print("\nValores registrados no LEIAME.txt:")
print("K-NN: 78.11% | K-Means k: 2 | silhueta: 0.1716")