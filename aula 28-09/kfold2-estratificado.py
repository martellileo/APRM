import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression

# X = [renda, score_credito]
X = np.array([
    [5000, 750],
    [4200, 720],
    [6000, 780],
    [3500, 690],
    [7000, 800],
    [4800, 710],
    [5500, 740],
    [3200, 650],
    [2500, 580],
    [2000, 520],
    [6200, 760],
    [3800, 680],
    [4500, 700],
    [2800, 610],
    [8000, 820]
])

# 0 = não inadimplente
# 1 = inadimplente
y = np.array([
    0, 0, 0, 0, 0,
    0, 0, 0, 1, 1,
    0, 0, 0, 1, 0
])

modelo = LogisticRegression(max_iter=1000)

dobras = StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=42
)

scores = cross_val_score(
    modelo,
    X,
    y,
    cv=dobras,
    scoring="accuracy"
)

print(scores)
print("Acurácia média:", scores.mean())

'''
X                            y

renda   score             inadimplente

5000     750       →           0
4200     720       →           0
6000     780       →           0
...
2500     580       →           1
2000     520       →           1
'''