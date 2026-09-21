"""
Aprendizado de Maquina - IFSP Presidente Epitacio
Aula 8, Parte 1 - DEMONSTRACAO GUIADA (45 min)

"""
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
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


def montar_preprocessador(escalar=True):
    """Imputacao + (escala) nos numericos; imputacao + one-hot na categorica.
    Tudo dentro de um ColumnTransformer: as estatisticas (mediana, media,
    desvio, categorias) sao aprendidas SO no treino."""
    passos_num = [("imp", SimpleImputer(strategy="median"))]
    if escalar:
        passos_num.append(("esc", StandardScaler()))
    return ColumnTransformer([
        ("num", Pipeline(passos_num), COL_NUM),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), COL_CAT),
    ])


def criar_modelos():
    return {
        "k-NN (k=5)": KNeighborsClassifier(n_neighbors=5),
        "Regressao Logistica": LogisticRegression(max_iter=1000),
        "Arvore de Decisao (max_depth=4)": DecisionTreeClassifier(max_depth=4, random_state=SEMENTE),
        "Naive Bayes (Gaussiano)": GaussianNB(),
        "SVM (kernel=rbf)": SVC(kernel="rbf", random_state=SEMENTE),
    }


def metricas(y_verd, y_prev):
    return {
        "acuracia": accuracy_score(y_verd, y_prev),
        "precisao": precision_score(y_verd, y_prev, zero_division=0),
        "recall": recall_score(y_verd, y_prev, zero_division=0),
        "f1": f1_score(y_verd, y_prev, zero_division=0),
    }


# ==================================================================== BLOCO 1
titulo(1, "O PROBLEMA E OS DADOS")

df = gerar_dados_emprestimo(n=450, semente=SEMENTE)
X = df.drop(columns="aprovado")
y = df["aprovado"]

print(f"Solicitacoes: {len(df)}  |  atributos: {X.shape[1]}  |  alvo: aprovado (0 = negado, 1 = aprovado)")
print("\nValores ausentes por coluna:")
print(df.isna().sum().to_string())
print("\nProporcao de cada classe:")
print(y.value_counts(normalize=True).round(3).to_string())
print("""
LEITURA: o problema e de CLASSIFICACAO binaria (aprovar ou negar um
emprestimo). Ha valores ausentes em tres colunas e uma coluna categorica
(cidade) - por isso os dados NAO podem ir direto para os algoritmos.
As duas classes estao quase equilibradas, entao a acuracia e uma metrica
razoavel para comecar.""")

# ==================================================================== BLOCO 2
titulo(2, "O PISO: UM MODELO QUE NAO APRENDE NADA")

X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.30, random_state=SEMENTE, stratify=y
)
print(f"Treino: {len(X_treino)} linhas  |  Teste: {len(X_teste)} linhas")
print(f"Proporcao de aprovados - treino: {y_treino.mean():.3f}  |  teste: {y_teste.mean():.3f}")

baseline = DummyClassifier(strategy="most_frequent").fit(X_treino, y_treino)
acc_base = accuracy_score(y_teste, baseline.predict(X_teste))
print(f"\nBaseline (sempre prever a classe mais comum no treino): acuracia = {acc_base:.3f}")
print("""
LEITURA: o baseline e o piso de qualquer comparacao. Um modelo que nao
supera "chutar sempre a classe mais comum" nao aprendeu nada util. O
parametro stratify=y garante que treino e teste tenham a MESMA proporcao
de aprovados - sem isso, o sorteio poderia entregar um teste torto.""")

# ==================================================================== BLOCO 3
titulo(3, "OS 5 MODELOS, MESMO PRE-PROCESSAMENTO, MESMO TREINO, MESMO TESTE")

resultados = {}
predicoes = {}
for nome, modelo in criar_modelos().items():
    pipe = Pipeline([("prep", montar_preprocessador()), ("modelo", modelo)])
    pipe.fit(X_treino, y_treino)
    y_prev = pipe.predict(X_teste)
    predicoes[nome] = y_prev
    resultados[nome] = metricas(y_teste, y_prev)

tabela = pd.DataFrame(resultados).T.round(3)
tabela.loc["Baseline (classe mais comum)"] = [acc_base, np.nan, 0.0, 0.0]
print(tabela.to_string())
print("""
LEITURA: todos os 5 modelos superam o baseline com folga - ha sinal nos
dados. As diferencas ENTRE os modelos sao pequenas (poucos pontos
percentuais). Com apenas 135 exemplos de teste, cada exemplo vale cerca de
0,7 ponto percentual - uma diferenca de 1 ou 2 pontos pode ser so sorte
do sorteio. Guarde essa ideia: ela volta no BLOCO 7.""")

fig, ax = plt.subplots(figsize=(8, 4))
ordem = tabela.drop(index="Baseline (classe mais comum)").sort_values("acuracia")
ax.barh(ordem.index, ordem["acuracia"], color="#2e7d32")
ax.axvline(acc_base, color="#c62828", linestyle="--", label=f"baseline = {acc_base:.3f}")
ax.set_xlim(0.4, 1.0)
ax.set_xlabel("Acuracia no conjunto de teste")
ax.set_title("5 classificadores no mesmo problema (um unico sorteio)")
ax.legend(loc="lower right")
for i, v in enumerate(ordem["acuracia"]):
    ax.text(v + 0.005, i, f"{v:.3f}", va="center")
plt.tight_layout()
plt.savefig("comparacao_modelos.png", dpi=150)
plt.close()
print("Figura salva: comparacao_modelos.png")

# ==================================================================== BLOCO 4
titulo(4, "ALEM DA ACURACIA: MATRIZ DE CONFUSAO, PRECISAO E RECALL")

print("Convencao: classe 1 = APROVADO (classe positiva).")
print("  precisao = dos que o modelo APROVOU, quantos realmente eram bons pagadores")
print("  recall   = dos que ERAM bons pagadores, quantos o modelo aprovou\n")

fig, eixos = plt.subplots(1, 5, figsize=(16, 3.3))
for ax, (nome, y_prev) in zip(eixos, predicoes.items()):
    cm = confusion_matrix(y_teste, y_prev)
    ax.imshow(cm, cmap="Greens")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=13)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["negado", "aprovado"]); ax.set_yticklabels(["negado", "aprovado"])
    ax.set_xlabel("previsto"); ax.set_ylabel("real")
    ax.set_title(nome.split(" (")[0], fontsize=10)
    tn, fp, fn, tp = cm.ravel()
    print(f"{nome:34s} aprovacoes indevidas (FP) = {fp:2d} | negacoes indevidas (FN) = {fn:2d}")
plt.tight_layout()
plt.savefig("matrizes_confusao.png", dpi=150)
plt.close()
print("Figura salva: matrizes_confusao.png")
print("""
LEITURA: dois modelos com acuracia parecida podem errar de jeitos
DIFERENTES. Em credito, os dois erros tem custos diferentes: aprovar quem
nao vai pagar (falso positivo) costuma custar mais que negar quem pagaria
(falso negativo). Quem escolhe o modelo precisa olhar para QUAL erro
importa mais - a acuracia sozinha nao diz isso.""")

# ==================================================================== BLOCO 5
titulo(5, "O QUE A ESCALA FAZ POR CADA MODELO")

print("Mesmo treino e teste, mas SEM padronizar os atributos numericos:\n")
linhas = []
for nome, modelo in criar_modelos().items():
    com = resultados[nome]["acuracia"]
    pipe = Pipeline([("prep", montar_preprocessador(escalar=False)), ("modelo", modelo)])
    pipe.fit(X_treino, y_treino)
    sem = accuracy_score(y_teste, pipe.predict(X_teste))
    linhas.append((nome, com, sem, sem - com))
esc = pd.DataFrame(linhas, columns=["modelo", "com_escala", "sem_escala", "diferenca"]).set_index("modelo").round(3)
print(esc.to_string())
print("""
LEITURA: k-NN e SVM dependem de DISTANCIA - sem padronizar, renda
(milhares) domina idade (dezenas) e o desempenho despenca (o SVM cai de 0,859
para 0,607, bem mais perto do baseline). Regressao logistica, arvore e Naive Bayes
praticamente nao mudam de acuracia: a arvore compara cada atributo com um
limiar, que acompanha a escala; o Naive Bayes modela cada atributo
separadamente; a regressao logistica se ajusta ao "peso" de cada atributo
(mas com atributos sem escala a otimizacao fica mais lenta e os
coeficientes deixam de ser comparaveis entre si). Regra pratica: padronize
SEMPRE que o modelo usar distancia; nos demais, padronizar nao machuca e
ajuda a interpretar.""")

# ==================================================================== BLOCO 6
titulo(6, "UM HIPERPARAMETRO EM ACAO: SIMPLES DEMAIS x COMPLEXO DEMAIS")

print("k-NN: variando o numero de vizinhos (k):")
linhas = []
for k in [1, 3, 5, 15, 31, 71]:
    pipe = Pipeline([("prep", montar_preprocessador()), ("modelo", KNeighborsClassifier(k))]).fit(X_treino, y_treino)
    linhas.append((k, pipe.score(X_treino, y_treino), pipe.score(X_teste, y_teste)))
print(pd.DataFrame(linhas, columns=["k", "acc_treino", "acc_teste"]).round(3).to_string(index=False))

print("\nArvore de decisao: variando a profundidade maxima:")
linhas = []
for d in [1, 2, 4, 8, 16, None]:
    pipe = Pipeline([("prep", montar_preprocessador()),
                     ("modelo", DecisionTreeClassifier(max_depth=d, random_state=SEMENTE))]).fit(X_treino, y_treino)
    linhas.append((str(d), pipe.score(X_treino, y_treino), pipe.score(X_teste, y_teste)))
print(pd.DataFrame(linhas, columns=["max_depth", "acc_treino", "acc_teste"]).round(3).to_string(index=False))
print("""
LEITURA: com k=1 o k-NN acerta 100% do treino (cada ponto e o proprio
vizinho) mas cai no teste; a arvore sem limite de profundidade tambem
decora o treino (acuracia 1.000) e perde no teste. E o mesmo padrao da
regressao polinomial de grau alto: modelo flexivel demais memoriza o
ruido. Repare tambem que k=71 tem a MELHOR acuracia de teste - mas
escolher k olhando para o teste seria trapaca, porque o teste deixaria de
ser "dado nunca visto". Como ajustar hiperparametros com honestidade e
o tema da aula seguinte.""")

# ==================================================================== BLOCO 7
titulo(7, "O PROBLEMA DE UM UNICO SORTEIO")

print("Repetindo o experimento com 10 sorteios diferentes de treino/teste:\n")
nomes = list(criar_modelos().keys())
acuracias = {n: [] for n in nomes}
vencedores = []
for semente in range(10):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.30, random_state=semente, stratify=y)
    acc_rodada = {}
    for nome, modelo in criar_modelos().items():
        pipe = Pipeline([("prep", montar_preprocessador()), ("modelo", modelo)]).fit(X_tr, y_tr)
        acc_rodada[nome] = pipe.score(X_te, y_te)
        acuracias[nome].append(acc_rodada[nome])
    vencedores.append(max(acc_rodada, key=acc_rodada.get))

resumo = pd.DataFrame({
    "media": {n: np.mean(v) for n, v in acuracias.items()},
    "desvio": {n: np.std(v) for n, v in acuracias.items()},
    "minimo": {n: np.min(v) for n, v in acuracias.items()},
    "maximo": {n: np.max(v) for n, v in acuracias.items()},
}).round(3)
print(resumo.to_string())
print("\nQuem foi o 1o colocado em cada um dos 10 sorteios:")
print(pd.Series(vencedores).value_counts().to_string())

fig, ax = plt.subplots(figsize=(9, 4))
ax.boxplot([acuracias[n] for n in nomes])
ax.set_xticklabels([n.split(" (")[0] for n in nomes])
ax.set_ylabel("Acuracia no teste")
ax.set_title("A mesma comparacao em 10 sorteios diferentes")
plt.setp(ax.get_xticklabels(), rotation=15)
plt.tight_layout()
plt.savefig("ranking_muda_com_sorteio.png", dpi=150)
plt.close()
print("Figura salva: ranking_muda_com_sorteio.png")
print("""
LEITURA: mudando so o sorteio, o "melhor modelo" muda, e as faixas de
desempenho dos modelos se sobrepoem bastante. Conclusao: uma unica divisao
treino/teste e uma estimativa RUIDOSA - declarar um vencedor com base nela
e arriscado. Como avaliar de forma mais confiavel (validacao cruzada,
metricas adequadas, curvas de aprendizado) e o assunto da proxima sessao.""")

print("""
=======================================================================
Demonstracao concluida. Figuras geradas: comparacao_modelos.png,
matrizes_confusao.png e ranking_muda_com_sorteio.png
=======================================================================""")
