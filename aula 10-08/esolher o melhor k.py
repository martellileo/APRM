from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from sklearn.datasets import load_iris

SEMENTE = 42

dados = load_iris()
X = dados.data
y = dados.target

X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.30, random_state=SEMENTE, stratify=y,
)

parametros = {
    "n_neighbors": [1, 3, 5, 7, 9, 11, 15, 21, 31, 51]
}

busca = GridSearchCV(
    KNeighborsClassifier(),
    parametros,
    cv=5,
    scoring="accuracy"
)

busca.fit(X_treino, y_treino)

print("Melhor k:", busca.best_params_["n_neighbors"])
print("Acurácia média na validação:", busca.best_score_)

melhor_modelo = busca.best_estimator_
previsoes = melhor_modelo.predict(X_teste)
acuracia_final = accuracy_score(
    y_teste,
    previsoes
)

print("Acurácia final no teste:", acuracia_final)

print("\nMatriz de confusão:")
print(confusion_matrix(y_teste, previsoes))

print("\nRelatório de classificação:")
print(classification_report(
    y_teste,
    previsoes,
    target_names=dados.target_names,
    digits=3
))
