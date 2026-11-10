from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression
import numpy as np

X = np.array([
    [50,  1],
    [60,  2],
    [70,  2],
    [80,  3],
    [90,  3],
    [100, 3],
    [110, 4],
    [120, 4],
    [130, 4],
    [150, 5]
])

# y = preço do imóvel (milhares de reais)
y = np.array([
    180, 210, 240, 280, 310,
    340, 380, 410, 450, 510
])

modelo = LinearRegression()
dobras = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scores = cross_val_score(
    modelo,
    X,
    y,
    cv=dobras,
    scoring="r2"
)

print(scores)
print("Média:", scores.mean())

'''
X                         y

50m²,  1 quarto    →     180 mil
60m²,  2 quartos   →     210 mil
70m²,  2 quartos   →     240 mil
...
150m², 5 quartos   →     510 mil
'''