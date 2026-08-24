import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report

# 1. Carregando o dataset sujo
df = pd.read_csv('bg3_personagens.csv')

# --- ETAPA DE TRATAMENTO DE DADOS ---
# Tratando valores nulos (preenchendo raça com a moda e nível/força com a mediana)
df['raca'] = df['raca'].str.strip().str.title() # Limpa espaços e padroniza texto
df['raca'] = df['raca'].fillna(df['raca'].mode()[0])
df['nivel'] = df['nivel'].fillna(df['nivel'].median())
df['forca'] = df['forca'].fillna(df['forca'].median())

# Removendo Outliers / Erros absurdos
df = df[(df['destreza'] >= 1) & (df['destreza'] <= 30)]
df = df[(df['nivel'] >= 1) & (df['nivel'] <= 20)]

# 2. Criando uma Variável Alvo para o KNN (Classificação)
# Exemplo: Vamos prever se o personagem é um Combatente Pesado (Guerreiro ou Paladino = 1, Outros = 0)
df['alvo_combatente'] = df['classe'].isin(['Guerreiro', 'Paladino']).astype(int)

# 3. Preparando os dados (Features e Target)
# Removemos colunas de texto descritivas e o alvo
X = df.drop(columns=['nome_personagem', 'classe', 'alvo_combatente'])

# Convertendo a coluna categórica 'raca' em variáveis numéricas (One-Hot Encoding)
X = pd.get_dummies(X, columns=['raca'], drop_first=True)

y = df['alvo_combatente']

# Divisão em Treino e Teste (com random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Padronização das escalas (fundamental para KNN e K-Means)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# 4. APLICANDO O KNN (Supervisionado)
# ==========================================
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

# Fazendo previsões
y_pred = knn.predict(X_test_scaled)

# Calculando a Acurácia
acuracia = accuracy_score(y_test, y_pred)
print(f"--- RESULTADO DO KNN ---")
print(f"Acurácia do modelo KNN: {acuracia * 100:.2f}%\n")
print("Relatório de Classificação:")
print(classification_report(y_test, y_pred))

# ==========================================
# 5. APLICANDO O K-MEANS (Não Supervisionado)
# ==========================================
# Vamos agrupar os personagens em 3 arquétipos com base em seus atributos físicos e mentais
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df.loc[X_train.index, 'cluster'] = kmeans.fit_predict(X_train_scaled)

print(f"--- RESULTADO DO K-MEANS ---")
print(f"K-Means executado com sucesso! Centróides criados para 3 clusters.")
print(df['cluster'].value_counts())