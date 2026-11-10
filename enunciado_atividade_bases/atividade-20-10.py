import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.metrics import classification_report, mean_squared_error
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# base 1:

df1 = pd.read_csv('base1_classificacao_churn.csv').drop(columns=['id_cliente'])
X1, y1 = df1.drop(columns=['cancelou']), df1['cancelou']
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2, random_state=42)

pre1 = ColumnTransformer([
    ('num', Pipeline([('imp', SimpleImputer(strategy='median')), ('scl', StandardScaler())]), X1.select_dtypes(include=np.number).columns),
    ('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')), ('enc', OneHotEncoder(handle_unknown='ignore'))]), X1.select_dtypes(exclude=np.number).columns)
])

print(f"Dummy Acc: {Pipeline([('pre', pre1), ('clf', DummyClassifier())]).fit(X1_train, y1_train).score(X1_test, y1_test):.2f}")

modelos1 = {'LogReg': LogisticRegression(), 'RF': RandomForestClassifier(random_state=42), 'GB': GradientBoostingClassifier(random_state=42)}
for nome, modelo in modelos1.items():
    pipe = Pipeline([('pre', pre1), ('clf', modelo)]).fit(X1_train, y1_train)
    print(f"{nome} Acc: {pipe.score(X1_test, y1_test):.2f}")

melhor_b1 = Pipeline([('pre', pre1), ('clf', GradientBoostingClassifier(random_state=42))]).fit(X1_train, y1_train)
print("\nRelatório do melhor (GB):\n", classification_report(y1_test, melhor_b1.predict(X1_test)))


# base 2:

df2 = pd.read_csv('base2_regressao_imoveis.csv')
X2, y2 = df2.drop(columns=['preco_venda']), df2['preco_venda']
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=42)

pre2 = ColumnTransformer([
    ('num', Pipeline([('imp', SimpleImputer(strategy='median')), ('scl', StandardScaler())]), X2.select_dtypes(include=np.number).columns),
    ('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')), ('enc', OneHotEncoder(handle_unknown='ignore'))]), X2.select_dtypes(exclude=np.number).columns)
])

dummy2 = Pipeline([('pre', pre2), ('reg', DummyRegressor())]).fit(X2_train, y2_train)
print(f"Dummy RMSE: R$ {np.sqrt(mean_squared_error(y2_test, dummy2.predict(X2_test))):.2f}")

modelos2 = {'LinReg': LinearRegression(), 'RF': RandomForestRegressor(random_state=42)}
for nome, modelo in modelos2.items():
    pipe = Pipeline([('pre', pre2), ('reg', modelo)]).fit(X2_train, y2_train)
    print(f"{nome} RMSE: R$ {np.sqrt(mean_squared_error(y2_test, pipe.predict(X2_test))):.2f}")


#base 3

df3 = pd.read_csv('base3_agrupamento_clientes.csv').drop(columns=['id_cliente', 'avaliacao_app'])
X3 = StandardScaler().fit_transform(df3.fillna(df3.median()))

for k in [2, 3, 4]:
    labels = KMeans(n_clusters=k, random_state=42, n_init='auto').fit_predict(X3)
    print(f"K={k} | Silhueta: {silhouette_score(X3, labels):.3f}")

df3['Cluster'] = KMeans(n_clusters=3, random_state=42, n_init='auto').fit_predict(X3)
print("\nPerfis Médios (K=3):")
print(df3.groupby('Cluster').mean().round(1))