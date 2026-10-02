import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import silhouette_score

df = pd.read_csv('bg3_personagens_sujo.csv')
print("Dimensões originais (Linhas x Colunas):", df.shape)

df['raca'] = df['raca'].str.strip().str.title()
df['raca'] = df['raca'].fillna(df['raca'].mode()[0])
df['nivel'] = df['nivel'].fillna(df['nivel'].median())
df['forca'] = df['forca'].fillna(df['forca'].median())

df = df[(df['destreza'] >= 1) & (df['destreza'] <= 30)]
df = df[(df['nivel'] >= 1) & (df['nivel'] <= 20)]

y = df['classe']

X = df.drop(columns=['nome_personagem', 'classe'])

X = pd.get_dummies(X, columns=['raca'], drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

y_pred = knn.predict(X_test_scaled)

acuracia = accuracy_score(y_test, y_pred)
print(f"--- RESULTADO DO KNN (11 CLASSES) ---")
print(f"Acurácia do modelo KNN: {acuracia * 100:.2f}%\n")
print("Relatório de Classificação:")
print(classification_report(y_test, y_pred))

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df.loc[X_train.index, 'cluster'] = kmeans.fit_predict(X_train_scaled)

silhueta = silhouette_score(X_train_scaled, kmeans.labels_)
print(f"Coeficiente de Silhueta: {silhueta:.4f}")

print(f"\n--- RESULTADO DO K-MEANS ---")
print(f"K-Means executado com sucesso! Centróides criados para 3 clusters.")
print(df['cluster'].value_counts())