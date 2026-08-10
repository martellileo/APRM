from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEMENTE = 42

dados = load_iris()
X = dados.data
y = dados.target

X_treino, X_teste, y_treino, y_teste = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=SEMENTE,
    stratify=y
)

# Pipeline: primeiro padroniza; depois aplica o KNN
'''
Padronizar os atributos pode mudar a ordenação dos vizinhos, pois altera a
contribuição de cada atributo para o cálculo da distância.
Um resumo prático:

Classificador	      Usar StandardScaler?
KNN	                  Sim
SVM/SVC	              Sim
Regressão logística	  Recomendado
Rede neural	          Sim
Árvore de decisão	  Não é necessário
Random Forest	      Não é necessário
Gradient Boosting	  Não é necessário
Naive Bayes	Depende da variante
'''
modelo = make_pipeline(
    StandardScaler(),
    KNeighborsClassifier()
)

# No pipeline, precisamos indicar a etapa e o parâmetro:
# kneighborsclassifier__n_neighbors
parametros = {
    "kneighborsclassifier__n_neighbors":
        [1, 3, 5, 7, 9, 11, 15, 21, 31, 51]
}

#Faz a busca pelo melhor k
busca = GridSearchCV(
    modelo,
    parametros,
    cv=5,
    scoring="accuracy"
)

busca.fit(X_treino, y_treino)
melhor_k = busca.best_params_[
    "kneighborsclassifier__n_neighbors"
]

print("Melhor k:", melhor_k)
print(
    "Acurácia média na validação:",
    round(busca.best_score_, 4)
)

melhor_modelo = busca.best_estimator_
previsoes = melhor_modelo.predict(X_teste)

acuracia_final = accuracy_score(y_teste, previsoes)

print(
    "Acurácia final no teste:",
    round(acuracia_final, 4)
)
