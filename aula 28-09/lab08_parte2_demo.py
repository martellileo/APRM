"""
Aprendizado de Maquina - IFSP Presidente Epitacio
Aula 8, Parte 2 - DEMONSTRACAO GUIADA (45 min)

Avaliacao de modelos e validacao: como saber, com honestidade, se um
modelo e bom - e como comparar modelos sem se enganar. Cada bloco mostra
UM erro comum de avaliacao e a pratica que o corrige.

    python lab08_parte2_demo.py

Gera: holdout_instavel.png, kfold_esquema.png, curva_complexidade.png,
      curva_aprendizado.png
"""
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from sklearn.compose import ColumnTransformer
from sklearn.datasets import make_classification
from sklearn.dummy import DummyClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import (GridSearchCV, KFold, StratifiedKFold,
                                     cross_val_score, learning_curve,
                                     train_test_split, validation_curve)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")
SEMENTE = 42
pd.set_option("display.width", 100)


def titulo(n, texto):
    print()
    print("=" * 70)
    print(f"BLOCO {n} - {texto}")
    print("=" * 70)


def gerar_dados_emprestimo(n=450, semente=42):
    """Mesmo dataset das aulas de preparacao de dados: solicitacoes de
    emprestimo, com valores ausentes, uma coluna categorica e outliers."""
    rng = np.random.default_rng(semente)

    idade = rng.normal(38, 12, n).clip(18, 75).round().astype(float)
    score_credito = rng.normal(650, 80, n).clip(300, 850).round().astype(float)
    tempo_emprego = rng.normal(7, 5, n).clip(0, 35).round().astype(float)

    renda = rng.lognormal(mean=8.15, sigma=0.45, size=n)
    idx_outlier = rng.choice(n, size=5, replace=False)
    renda[idx_outlier] *= rng.uniform(6, 10, size=5)
    renda = renda.round(2)

    cidades = np.array(["SP", "RJ", "MG", "Outra"])
    cidade = rng.choice(cidades, size=n, p=[0.42, 0.23, 0.15, 0.20])

    renda_c = np.clip(renda, None, np.percentile(renda, 95))
    z = (
        0.05 * (score_credito - 650)
        + 0.0008 * (renda_c - renda_c.mean())
        + 0.12 * (tempo_emprego - 7)
        + rng.normal(0, 0.35, n)
    )
    prob_aprovado = 1 / (1 + np.exp(-z))
    aprovado = (rng.uniform(0, 1, n) < prob_aprovado).astype(int)

    df = pd.DataFrame({
        "idade": idade,
        "renda": renda,
        "tempo_emprego": tempo_emprego,
        "score_credito": score_credito,
        "cidade": cidade,
        "aprovado": aprovado,
    })

    mask_idade = rng.uniform(0, 1, n) < 0.08
    mask_renda = rng.uniform(0, 1, n) < 0.10
    mask_cidade = rng.uniform(0, 1, n) < 0.06
    df.loc[mask_idade, "idade"] = np.nan
    df.loc[mask_renda, "renda"] = np.nan
    df.loc[mask_cidade, "cidade"] = None
    return df


COL_NUM = ["idade", "renda", "tempo_emprego", "score_credito"]
COL_CAT = ["cidade"]


def montar_preprocessador():
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("esc", StandardScaler())]), COL_NUM),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), COL_CAT),
    ])


def com_prep(modelo):
    return Pipeline([("prep", montar_preprocessador()), ("modelo", modelo)])


df = gerar_dados_emprestimo(n=450, semente=SEMENTE)
X = df.drop(columns="aprovado")
y = df["aprovado"]
validacao = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMENTE)

# ==================================================================== BLOCO 1
titulo(1, "HOLDOUT: UMA UNICA DIVISAO E UMA ESTIMATIVA INSTAVEL")

print("Mesmo modelo (regressao logistica), 200 divisoes treino/teste diferentes:\n")
acuracias = []
for semente in range(200):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.30, random_state=semente, stratify=y)
    acuracias.append(com_prep(LogisticRegression(max_iter=1000)).fit(X_tr, y_tr).score(X_te, y_te))
acuracias = np.array(acuracias)
print(f"  media   = {acuracias.mean():.3f}")
print(f"  desvio  = {acuracias.std():.3f}")
print(f"  minimo  = {acuracias.min():.3f}   (divisao 'azarada')")
print(f"  maximo  = {acuracias.max():.3f}   (divisao 'sortuda')")

fig, ax = plt.subplots(figsize=(8, 3.6))
ax.hist(acuracias, bins=18, color="#2e7d32", edgecolor="white")
ax.axvline(acuracias.mean(), color="#c62828", linestyle="--", label=f"media = {acuracias.mean():.3f}")
ax.set_xlabel("Acuracia no teste")
ax.set_ylabel("Numero de divisoes")
ax.set_title("Regressao logistica: 200 divisoes holdout diferentes")
ax.legend()
plt.tight_layout()
plt.savefig("holdout_instavel.png", dpi=150)
plt.close()
print("Figura salva: holdout_instavel.png")
print("""
LEITURA: e o MESMO modelo nos MESMOS dados - so mudou o sorteio de quem
foi para o teste. Quem tivesse feito uma unica divisao poderia reportar
qualquer valor entre o minimo e o maximo acima, e nenhum deles estaria
"errado". O holdout continua util (e simples, e rapido), mas uma unica
divisao nao mostra a incerteza da estimativa.""")

# ==================================================================== BLOCO 2
titulo(2, "VALIDACAO CRUZADA K-FOLD: TODO EXEMPLO VIRA TESTE UMA VEZ")

print("Exemplo minusculo: 10 exemplos, k = 5 dobras (KFold sem embaralhar)\n")
exemplos = np.arange(10)
for i, (idx_tr, idx_val) in enumerate(KFold(n_splits=5).split(exemplos), start=1):
    print(f"  dobra {i}: validacao = {idx_val}   treino = {idx_tr}")
print("""
Cada exemplo aparece em validacao exatamente UMA vez e em treino nas
outras k-1 vezes. O desempenho final e a MEDIA dos k desempenhos.
""")

print("Na base de emprestimos (450 linhas), com StratifiedKFold(5):")
print("  A proporcao de aprovados em cada dobra de validacao:")
for i, (idx_tr, idx_val) in enumerate(validacao.split(X, y), start=1):
    print(f"  dobra {i}: {len(idx_val)} exemplos | aprovados = {y.iloc[idx_val].mean():.3f}")
print(f"  proporcao geral = {y.mean():.3f}")

fig, ax = plt.subplots(figsize=(8, 2.6))
for i in range(5):
    for j in range(5):
        cor = "#c62828" if i == j else "#2e7d32"
        ax.add_patch(Rectangle((j, 4 - i), 0.95, 0.8, color=cor))
    ax.text(-0.1, 4 - i + 0.4, f"iteracao {i + 1}", ha="right", va="center", fontsize=9)
for j in range(5):
    ax.text(j + 0.475, 5.05, f"dobra {j + 1}", ha="center", fontsize=9)
ax.set_xlim(-1.6, 5); ax.set_ylim(-0.9, 5.5); ax.axis("off")
ax.text(5, -0.55, "verde = treino   |   vermelho = validacao", ha="right", fontsize=9)
plt.tight_layout()
plt.savefig("kfold_esquema.png", dpi=150)
plt.close()
print("Figura salva: kfold_esquema.png")
print("""
LEITURA: "estratificado" significa que cada dobra preserva a proporcao de
cada classe - importante em classificacao, principalmente quando uma
classe e rara. shuffle=True embaralha antes de dividir, para que a ordem
das linhas do arquivo nao vire vies.""")

# ==================================================================== BLOCO 3
titulo(3, "COMPARANDO MODELOS COM AS MESMAS DOBRAS: MEDIA E DESVIO")

modelos = {
    "k-NN (k=5)": KNeighborsClassifier(n_neighbors=5),
    "Regressao Logistica": LogisticRegression(max_iter=1000),
    "Arvore (max_depth=4)": DecisionTreeClassifier(max_depth=4, random_state=SEMENTE),
    "Naive Bayes": GaussianNB(),
    "SVM (rbf)": SVC(kernel="rbf", random_state=SEMENTE),
}
linhas = []
for nome, modelo in modelos.items():
    notas = cross_val_score(com_prep(modelo), X, y, cv=validacao)
    linhas.append((nome, notas.mean(), notas.std(), notas.min(), notas.max()))
tab = pd.DataFrame(linhas, columns=["modelo", "media", "desvio", "minimo", "maximo"]).set_index("modelo").round(3)
print(tab.sort_values("media", ascending=False).to_string())
print("""
LEITURA: todos os modelos foram avaliados nas MESMAS 5 dobras (mesmo
objeto de validacao) - comparacao justa. Repare no DESVIO: as medias de
SVM e regressao logistica diferem por menos que 1 desvio de qualquer um
dos dois. Quando a diferenca entre as medias e menor que a variacao entre
as dobras, os modelos sao, na pratica, EMPATADOS - nao declare um vencedor
so porque uma media ficou 0,004 acima. Ja a distancia entre SVM (0,860) e
k-NN (0,798) - cerca de 0,06, mais que um desvio - e um sinal bem mais
forte de que existe diferenca real.""")

# ==================================================================== BLOCO 4
titulo(4, "SUBAJUSTE x SOBREAJUSTE: A CURVA DE COMPLEXIDADE")

ks = [1, 3, 5, 9, 15, 25, 45, 75, 125, 200, 300]
tr_k, va_k = validation_curve(com_prep(KNeighborsClassifier()), X, y,
                              param_name="modelo__n_neighbors", param_range=ks, cv=validacao)
profs = list(range(1, 13))
tr_d, va_d = validation_curve(com_prep(DecisionTreeClassifier(random_state=SEMENTE)), X, y,
                              param_name="modelo__max_depth", param_range=profs, cv=validacao)

print("k-NN - k pequeno = modelo COMPLEXO; k grande = modelo SIMPLES:")
print(pd.DataFrame({"k": ks, "treino": tr_k.mean(1), "validacao": va_k.mean(1)}).round(3).to_string(index=False))
print("\nArvore - profundidade pequena = SIMPLES; profundidade grande = COMPLEXA:")
print(pd.DataFrame({"max_depth": profs, "treino": tr_d.mean(1), "validacao": va_d.mean(1)}).round(3).to_string(index=False))

fig, eixos = plt.subplots(1, 2, figsize=(11, 3.8))
for ax, xs, tr, va, xlabel, log in [
    (eixos[0], ks, tr_k, va_k, "k (numero de vizinhos)", True),
    (eixos[1], profs, tr_d, va_d, "max_depth (profundidade da arvore)", False),
]:
    ax.plot(xs, tr.mean(1), "o-", color="#1565c0", label="treino")
    ax.plot(xs, va.mean(1), "o-", color="#c62828", label="validacao (CV)")
    ax.set_xlabel(xlabel); ax.set_ylabel("Acuracia"); ax.legend()
    if log:
        ax.set_xscale("log")
eixos[0].set_title("k-NN: k grande = simples, k pequeno = complexo")
eixos[1].set_title("Arvore: profundidade grande = complexa")
plt.tight_layout()
plt.savefig("curva_complexidade.png", dpi=150)
plt.close()
print("Figura salva: curva_complexidade.png")
print("""
LEITURA: no k-NN, k=1 acerta 100% do treino mas so ~74% na validacao
(SOBREAJUSTE: grande distancia entre as curvas). No outro extremo, k=300
usa mais de 80% do treino como vizinhos e as DUAS curvas caem (SUBAJUSTE:
o modelo e simples demais ate para o treino). O melhor k fica no meio,
onde a curva de validacao e mais alta. Na arvore, a curva de treino sobe
sem parar (chega a ~1,0) enquanto a de validacao cai: cada nivel a mais
decora mais ruido.""")

# ==================================================================== BLOCO 5
titulo(5, "CURVA DE APRENDIZADO: MAIS DADOS AJUDAM?")

tamanhos = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
fig, eixos = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
for ax, (nome, modelo) in zip(eixos, [
    ("Regressao logistica (simples)", LogisticRegression(max_iter=1000)),
    ("Arvore sem limite (complexa)", DecisionTreeClassifier(random_state=SEMENTE)),
]):
    n_ex, tr, va = learning_curve(com_prep(modelo), X, y, train_sizes=tamanhos, cv=validacao)
    print(f"{nome}:")
    print(pd.DataFrame({"n_treino": n_ex, "treino": tr.mean(1), "validacao": va.mean(1)}).round(3).to_string(index=False))
    ax.plot(n_ex, tr.mean(1), "o-", color="#1565c0", label="treino")
    ax.plot(n_ex, va.mean(1), "o-", color="#c62828", label="validacao (CV)")
    ax.set_title(nome); ax.set_xlabel("Exemplos usados no treino"); ax.legend()
eixos[0].set_ylabel("Acuracia")
plt.tight_layout()
plt.savefig("curva_aprendizado.png", dpi=150)
plt.close()
print("Figura salva: curva_aprendizado.png")
print("""
LEITURA: na regressao logistica, treino e validacao CONVERGEM para
valores proximos (~0,86) - a lacuna e pequena e, a partir de ~200
exemplos, mais dados quase nao ajudam: o limite esta no modelo, nao na
quantidade de dados. Na arvore sem limite a lacuna e ENORME (treino = 1,0
sempre; validacao entre 0,73 e 0,80) e so diminui devagar com mais dados:
o modelo tem variancia alta. Cada
curva aponta para uma cura diferente (modelo mais rico x mais dados ou
modelo mais simples).""")

# ==================================================================== BLOCO 6
titulo(6, "ACURACIA ENGANA: CLASSES DESBALANCEADAS")

X_r, y_r = make_classification(n_samples=1000, n_features=6, n_informative=4, n_redundant=0,
                               n_clusters_per_class=1, weights=[0.95, 0.05], flip_y=0.0,
                               class_sep=1.0, random_state=SEMENTE)
print(f"Problema tipo 'deteccao de fraude': {len(y_r)} exemplos, {y_r.mean() * 100:.1f}% sao a classe rara (1)")
X_tr, X_te, y_tr, y_te = train_test_split(X_r, y_r, test_size=0.30, random_state=SEMENTE, stratify=y_r)

candidatos = {
    "Baseline (sempre 'normal')": DummyClassifier(strategy="most_frequent"),
    "Regressao Logistica": LogisticRegression(max_iter=1000),
    "Reg. Logistica (class_weight='balanced')": LogisticRegression(max_iter=1000, class_weight="balanced"),
}
linhas = []
for nome, modelo in candidatos.items():
    modelo.fit(X_tr, y_tr)
    y_prev = modelo.predict(X_te)
    tn, fp, fn, tp = confusion_matrix(y_te, y_prev).ravel()
    linhas.append((nome, accuracy_score(y_te, y_prev), precision_score(y_te, y_prev, zero_division=0),
                   recall_score(y_te, y_prev), f1_score(y_te, y_prev), fn, fp))
res = pd.DataFrame(linhas, columns=["modelo", "acuracia", "precisao", "recall", "f1",
                                     "fraudes_perdidas(FN)", "alarmes_falsos(FP)"]).set_index("modelo").round(3)
print(res.to_string())
print("""
LEITURA: o baseline "nunca ha fraude" tem 95% de acuracia sem aprender
NADA - e recall zero: nenhuma fraude detectada. A regressao logistica
comum sobe para ~97,7% de acuracia, o que parece excelente, mas ainda
deixa passar 40% das fraudes (recall 0,60). Com class_weight="balanced"
a acuracia CAI (~94%, ate abaixo do baseline!), porem o recall vai a ~93%:
o modelo passa a pegar quase todas as fraudes ao custo de mais alarmes
falsos. Qual e "melhor" depende do CUSTO de cada erro - a acuracia sozinha
nao responde. Em desbalanceamento, olhe matriz de confusao, precisao,
recall e F1 da classe rara.""")

# ==================================================================== BLOCO 7
titulo(7, "VAZAMENTO DE DADOS: O ERRO QUE FAZ O MODELO PARECER MAGICO")

print("Cenario: 60 exemplos, 2000 atributos de PURO RUIDO e alvo sorteado (0/1).")
print("Nao existe padrao nenhum a aprender - a acuracia honesta e ~0,50.")
print("Repetindo em 20 conjuntos de ruido diferentes:\n")
errado, certo = [], []
for semente in range(20):
    rng = np.random.default_rng(semente)
    X_ruido = rng.normal(size=(60, 2000))
    y_ruido = rng.integers(0, 2, 60)
    dobras = StratifiedKFold(5, shuffle=True, random_state=SEMENTE)

    # ERRADO: seleciona os 20 melhores atributos usando TODOS os dados
    # (inclusive as linhas que depois serao usadas como validacao)
    X_sel = SelectKBest(f_classif, k=20).fit_transform(X_ruido, y_ruido)
    errado.append(cross_val_score(LogisticRegression(max_iter=1000), X_sel, y_ruido, cv=dobras).mean())

    # CERTO: a selecao de atributos acontece DENTRO de cada dobra, via Pipeline
    pipe = Pipeline([("sel", SelectKBest(f_classif, k=20)), ("modelo", LogisticRegression(max_iter=1000))])
    certo.append(cross_val_score(pipe, X_ruido, y_ruido, cv=dobras).mean())
print(f"  selecao ANTES de dividir (vazamento) : acuracia media = {np.mean(errado):.3f}")
print(f"  selecao DENTRO do pipeline (correto) : acuracia media = {np.mean(certo):.3f}")
print("""
LEITURA: o processo "errado" reporta ~91% de acuracia num problema onde
nao ha nada a aprender! A selecao de atributos olhou o alvo de TODAS as
linhas, inclusive das que depois foram usadas como validacao - a
informacao do teste "vazou" para o treino. O mesmo vale para imputacao,
padronizacao e qualquer passo que aprende algo dos dados: deve ser
ajustado SO com o treino de cada dobra. Por isso usamos Pipeline com o
pre-processamento dentro - como em todos os blocos anteriores.""")

# ==================================================================== BLOCO 8
titulo(8, "ESCOLHER HIPERPARAMETROS SEM TOCAR NO TESTE")

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=SEMENTE, stratify=y)
busca = GridSearchCV(com_prep(KNeighborsClassifier()),
                     param_grid={"modelo__n_neighbors": [1, 3, 5, 9, 15, 25, 45]},
                     cv=validacao)
busca.fit(X_tr, y_tr)
resultados = pd.DataFrame(busca.cv_results_)[["param_modelo__n_neighbors", "mean_test_score", "std_test_score"]]
resultados.columns = ["k", "media_cv", "desvio_cv"]
print("Busca por k usando SO o conjunto de treino (validacao cruzada interna):")
print(resultados.round(3).to_string(index=False))
print(f"\nMelhor k pela validacao cruzada: {busca.best_params_['modelo__n_neighbors']}")
print(f"Acuracia media na validacao cruzada: {busca.best_score_:.3f}")
print(f"Acuracia no TESTE (usado uma unica vez, no final): {busca.score(X_te, y_te):.3f}")
print("""
LEITURA: o fluxo honesto tem TRES papeis para os dados: TREINO (ajusta
o modelo), VALIDACAO (aqui, as dobras internas - escolhe hiperparametros)
e TESTE (guardado ate o fim, usado UMA vez para reportar). Se voce olha
o teste para escolher k e depois reporta o mesmo teste, a nota vira
otimista - o teste deixou de ser "dado nunca visto".""")

print("""
=======================================================================
Demonstracao concluida. Figuras geradas: holdout_instavel.png,
kfold_esquema.png, curva_complexidade.png e curva_aprendizado.png
=======================================================================""")
