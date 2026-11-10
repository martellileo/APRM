"""
Aprendizado de Maquina - IFSP Presidente Epitacio
Laboratorio - Implementacao de um neuronio artificial e comparacao de funcoes de ativacao
***ARQUIVO DO ALUNO***

NOME: ______________________________  PRONTUARIO: ______________  DATA: __/__/____

INSTRUCOES
  - Preencha os trechos marcados com  # TODO
  - Use SEMPRE random_state=SEMENTE (42) e np.random.default_rng(SEMENTE)
  - Os exercicios usam as funcoes e a classe que voce escrever no exercicio 1:
    confira-as bem antes de seguir
  - Nao use bibliotecas de redes neurais (TensorFlow, PyTorch, MLPClassifier):
    o neuronio deve ser seu, em NumPy. O scikit-learn so aparece no exercicio 6
  - Rode o arquivo ate ele executar sem erro:   python lab_neuronio_exercicios.py
  - Responda as perguntas de interpretacao no bloco RELATORIO, no fim do arquivo
  - Entregue este arquivo executado

VALE 10,0 PONTOS
  codigo 40%  |  relatorio 60%
"""
import itertools
import warnings

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
SEMENTE = 42
pd.set_option("display.width", 110)


def secao(n, titulo):
    print()
    print("=" * 70)
    print(f"EXERCICIO {n} - {titulo}")
    print("=" * 70)


# ============================================================ EXERCICIO 1 (1,5)
secao(1, "O NEURONIO EM NUMPY")

# TODO 1a: escreva as funcoes degrau, identidade, sigmoide, tanh, relu e
#          leaky_relu (alpha=0.01). Todas devem aceitar escalar OU vetor NumPy.
#          Depois monte o dicionario ATIVACOES = {"degrau": degrau, ...}
def degrau(z):
    return np.where(np.asarray(z, dtype=float) >= 0, 1.0, 0.0)


def identidade(z):
    return np.asarray(z, dtype=float)


def sigmoide(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))


def tanh(z):
    return np.tanh(np.asarray(z, dtype=float))


def relu(z):
    return np.maximum(0.0, np.asarray(z, dtype=float))


def leaky_relu(z, alpha=0.01):
    z = np.asarray(z, dtype=float)
    return np.where(z > 0, z, alpha * z)


ATIVACOES = {
    "degrau": degrau,
    "identidade": identidade,
    "sigmoide": sigmoide,
    "tanh": tanh,
    "relu": relu,
    "leaky_relu": leaky_relu,
}

# TODO 1b: escreva a classe Neuronio(w, b, ativacao) com os metodos
#            z(X)     -> soma ponderada + vies (X pode ser um vetor (d,) ou
#                        um lote (n, d); use X @ w + b)
#            saida(X) -> a ativacao (buscada em ATIVACOES pelo nome) aplicada a z
#          Confira com x = (3; 1; -2), w = (0,2; -0,5; 0,6), b = 0,1:
#          imprima z e a saida de CADA uma das seis ativacoes
class Neuronio:
    def __init__(self, w, b, ativacao="sigmoide"):
        self.w = np.asarray(w, dtype=float)
        self.b = float(b)
        self.ativacao = ativacao

    def z(self, X):
        return np.asarray(X, dtype=float) @ self.w + self.b

    def saida(self, X):
        return ATIVACOES[self.ativacao](self.z(X))


x_1b = np.array([3.0, 1.0, -2.0])
w_1b = np.array([0.2, -0.5, 0.6])
b_1b = 0.1
print(f"1b) z = {Neuronio(w_1b, b_1b).z(x_1b):+.4f}")
for nome in ATIVACOES:
    print(f"  {nome:<11} y = {Neuronio(w_1b, b_1b, nome).saida(x_1b):+.4f}")

# TODO 1c: com o lote abaixo e a ativacao tanh, calcule as saidas de uma vez
#          (lote inteiro) e por um laco, amostra por amostra. Imprima z, y e
#          o resultado de np.allclose(lote, laco)
lote = np.array([[3, 1, -2], [0, 0, 0], [-1, 2, 1], [4, -3, 0.5]], dtype=float)
n_tanh = Neuronio(w_1b, b_1b, "tanh")
z_lote = n_tanh.z(lote)
y_lote = n_tanh.saida(lote)
y_laco = np.array([n_tanh.saida(amostra) for amostra in lote])
print(f"\n1c) z (lote) = {np.round(z_lote, 4)}")
print(f"    y (lote) = {np.round(y_lote, 4)}")
print(f"    y (laco) = {np.round(y_laco, 4)}")
print(f"    np.allclose(lote, laco) = {np.allclose(y_lote, y_laco)}")

# TODO 1d: confira sigmoide e tanh com os valores de referencia da apostila de
#          conceitos (use np.allclose com atol=5e-4 e imprima True/False)
zs_ref = np.array([-4, -2, -1, 0, 1, 2, 4], dtype=float)
REF_SIGMOIDE = np.array([0.018, 0.119, 0.269, 0.500, 0.731, 0.881, 0.982])
REF_TANH = np.array([-0.999, -0.964, -0.762, 0.000, 0.762, 0.964, 0.999])
print(f"\n1d) confere sigmoide: {np.allclose(sigmoide(zs_ref), REF_SIGMOIDE, atol=5e-4)}")
print(f"    confere tanh    : {np.allclose(tanh(zs_ref), REF_TANH, atol=5e-4)}")


# ============================================================ EXERCICIO 2 (1,5)
secao(2, "COMPARANDO AS ATIVACOES NAS MESMAS ENTRADAS")

# TODO 2a: com zs = np.linspace(-6, 6, 13), monte um DataFrame tab2 (indice z,
#          uma coluna por ativacao) e imprima com 3 casas decimais
zs = np.linspace(-6, 6, 13)
tab2 = pd.DataFrame({nome: fn(zs) for nome, fn in ATIVACOES.items()}, index=pd.Index(zs, name="z"))
print("2a) Tabela de ativacoes:")
print(tab2.round(3).to_string())

# TODO 2b: escreva as derivadas d_sigmoide, d_tanh, d_relu, d_leaky_relu,
#          d_degrau (zero) e d_identidade (um) e monte DERIVADAS = {...}.
#          Confira as quatro primeiras contra a derivada NUMERICA
#          (f(z+h) - f(z-h)) / (2h), h = 1e-5, em z_teste (sem o z = 0, onde a
#          ReLU nao tem derivada). Imprima a maior diferenca de cada uma
z_teste = np.array([-3.0, -1.0, -0.5, 0.5, 1.0, 3.0])


def d_degrau(z):
    return np.zeros_like(np.asarray(z, dtype=float))


def d_identidade(z):
    return np.ones_like(np.asarray(z, dtype=float))


def d_sigmoide(z):
    s = sigmoide(z)
    return s * (1.0 - s)


def d_tanh(z):
    return 1.0 - np.tanh(np.asarray(z, dtype=float)) ** 2


def d_relu(z):
    return np.where(np.asarray(z, dtype=float) > 0, 1.0, 0.0)


def d_leaky_relu(z, alpha=0.01):
    return np.where(np.asarray(z, dtype=float) > 0, 1.0, alpha)


DERIVADAS = {
    "degrau": d_degrau,
    "identidade": d_identidade,
    "sigmoide": d_sigmoide,
    "tanh": d_tanh,
    "relu": d_relu,
    "leaky_relu": d_leaky_relu,
}

h = 1e-5
print("\n2b) Maior diferenca absoluta contra derivada numerica:")
for nome in ["sigmoide", "tanh", "relu", "leaky_relu"]:
    f = ATIVACOES[nome]
    d_num = (f(z_teste + h) - f(z_teste - h)) / (2 * h)
    erro_max = np.max(np.abs(DERIVADAS[nome](z_teste) - d_num))
    print(f"  {nome:<11} max |d_analitica - d_numerica| = {erro_max:.2e}")

# TODO 2c: com fina = np.linspace(-10, 10, 2001), monte uma tabela com, para
#          degrau, sigmoide, tanh, relu e leaky_relu: minimo, maximo e media
#          da saida, derivada maxima e a FRACAO de pontos com derivada < 0,01
#          (fracao "saturada"). Imprima com 3 casas
fina = np.linspace(-10, 10, 2001)
linhas_2c = []
for nome in ["degrau", "sigmoide", "tanh", "relu", "leaky_relu"]:
    y_f = ATIVACOES[nome](fina)
    dy_f = DERIVADAS[nome](fina)
    linhas_2c.append({
        "ativacao": nome,
        "min": y_f.min(),
        "max": y_f.max(),
        "media": y_f.mean(),
        "deriv_max": dy_f.max(),
        "frac_saturada": np.mean(dy_f < 0.01),
    })
tab2c = pd.DataFrame(linhas_2c).set_index("ativacao")
print("\n2c) Resumo em [-10, 10]:")
print(tab2c.round(3).to_string())


# ============================================================ EXERCICIO 3 (1,5)
secao(3, "PORTAS LOGICAS, BUSCA EM GRADE E ESCALA DOS PESOS")

# TODO 3a: encontre a mao w e b (neuronio de degrau) para tres portas:
#            NOT   (1 entrada: saida 1 quando a entrada e 0)
#            AND3  (3 entradas: 1 so quando as tres sao 1)
#            MAJ3  (3 entradas: 1 quando pelo menos DUAS sao 1)
#          Para cada uma, percorra itertools.product([0, 1], repeat=n), imprima
#          as saidas e confira com a regra em Python (all(...), sum(...) >= 2)
portas_3a = {
    "NOT": ([-1.0], 0.5, 1, lambda e: int(not e[0])),
    "AND3": ([1.0, 1.0, 1.0], -2.5, 3, lambda e: int(all(e))),
    "MAJ3": ([1.0, 1.0, 1.0], -1.5, 3, lambda e: int(sum(e) >= 2)),
}
print("3a) Portas logicas com degrau:")
for nome, (w_p, b_p, n_ent, regra) in portas_3a.items():
    neu = Neuronio(w_p, b_p, "degrau")
    ents = list(itertools.product([0, 1], repeat=n_ent))
    saidas = [int(neu.saida(e)) for e in ents]
    esperadas = [regra(e) for e in ents]
    print(f"  {nome:<4} (w={w_p}, b={b_p}): saidas={saidas} | confere={saidas == esperadas}")

# TODO 3b: com grade = np.arange(-2, 2.01, 0.5), teste TODAS as combinacoes
#          (w1, w2, b) com degrau e conte quantas reproduzem AND, OR, XOR e
#          XNOR nas entradas (0,0), (0,1), (1,0), (1,1). Imprima as contagens
grade = np.arange(-2, 2.01, 0.5)
entradas2 = list(itertools.product([0, 1], repeat=2))
alvos_3b = {
    "AND": [0, 0, 0, 1],
    "OR": [0, 1, 1, 1],
    "XOR": [0, 1, 1, 0],
    "XNOR": [1, 0, 0, 1],
}
contagens_3b = {k: 0 for k in alvos_3b}
E2 = np.array(entradas2, dtype=float)
for w1, w2, b in itertools.product(grade, grade, grade):
    s = Neuronio([w1, w2], b, "degrau").saida(E2).astype(int).tolist()
    for k, alvo in alvos_3b.items():
        if s == alvo:
            contagens_3b[k] += 1
print(f"\n3b) Contagens na grade ({len(grade)**3} combinacoes): {contagens_3b}")

# TODO 3c: porta AND com w = (1; 1), b = -1,5 e ativacao SIGMOIDE. Imprima as
#          quatro saidas com os pesos e o vies multiplicados por k = 1 e por
#          k = 10 (4 casas decimais)
print("\n3c) Porta AND com sigmoide e escala k:")
for k in [1, 10]:
    y_k = Neuronio(k * np.array([1.0, 1.0]), k * (-1.5), "sigmoide").saida(E2)
    print(f"  k = {k:>2}: {[round(float(v), 4) for v in y_k]}")


# ============================================================ EXERCICIO 4 (1,5)
secao(4, "DUAS CAMADAS: XOR E O COLAPSO SEM NAO LINEARIDADE")


def camada(X, W, b, ativacao):
    """Camada densa: W tem formato (neuronios, entradas); b, (neuronios,)."""
    return ATIVACOES[ativacao](np.asarray(X, dtype=float) @ W.T + b)


E = np.array(entradas2, dtype=float)

# TODO 4a: monte uma rede de duas camadas com degrau que calcule o XOR:
#          oculta com 2 neuronios (um OR e um NAND), saida com 1 neuronio (AND
#          das duas saidas ocultas). Defina W1, b1, W2, b2, imprima a saida da
#          camada oculta e a saida final para as quatro entradas
W1 = np.array([[1.0, 1.0], [-1.0, -1.0]])
b1 = np.array([-0.5, 1.5])
W2 = np.array([[1.0, 1.0]])
b2 = np.array([-1.5])
H_4a = camada(E, W1, b1, "degrau")
Y_4a = camada(H_4a, W2, b2, "degrau")
print("4a) XOR com degrau (oculta OR e NAND; saida AND):")
print("  Camada oculta H:\n", H_4a.astype(int))
print("  Saida final Y  :", Y_4a.ravel().astype(int).tolist())

# TODO 4b: repita o calculo trocando TODAS as ativacoes por "identidade".
#          Imprima as quatro saidas, a rede colapsada W' = W2 @ W1 e
#          b' = W2 @ b1 + b2, e confira com np.allclose que E @ W'.T + b'
#          reproduz as saidas das duas camadas
H_4b = camada(E, W1, b1, "identidade")
Y_4b = camada(H_4b, W2, b2, "identidade")
W_col = W2 @ W1
b_col = W2 @ b1 + b2
Y_col = E @ W_col.T + b_col
print("\n4b) Com ativacao identidade nas duas camadas:")
print("  Saidas (2 camadas):", Y_4b.ravel().tolist())
print(f"  W' = W2 @ W1 = {W_col.tolist()}, b' = W2 @ b1 + b2 = {b_col.tolist()}")
print(f"  np.allclose(duas_camadas, colapsada) = {np.allclose(Y_4b, Y_col)}")

# TODO 4c: agora com ReLU na camada oculta e identidade na saida, resolva o XOR
#          com W1 = [[1, 1], [1, 1]], b1 = [0, -1] e escolha W2 (formato (1, 2))
#          e b2 para que a saida seja [0, 1, 1, 0]. Imprima a camada oculta e
#          a saida
W1_4c = np.array([[1.0, 1.0], [1.0, 1.0]])
b1_4c = np.array([0.0, -1.0])
W2_4c = np.array([[1.0, -2.0]])
b2_4c = np.array([0.0])
H_4c = camada(E, W1_4c, b1_4c, "relu")
Y_4c = camada(H_4c, W2_4c, b2_4c, "identidade")
print("\n4c) XOR com ReLU na oculta e identidade na saida:")
print("  Camada oculta H:\n", H_4c)
print("  Saida final Y  :", Y_4c.ravel().tolist())


# ============================================================ EXERCICIO 5 (1,5)
secao(5, "SATURACAO, GRADIENTE E RELU MORTA")

# TODO 5a: escreva cadeia(x0, w, L, ativacao): L camadas de UM neuronio cada,
#          a_l = phi(w * a_(l-1)) (sem vies), e devolva (a_L, d a_L / d x0),
#          em que a derivada e o produto de w * phi'(z_l) em todas as camadas
#          (use DERIVADAS do exercicio 2)
def cadeia(x0, w, L, ativacao):
    a = float(x0)
    grad = 1.0
    for _ in range(L):
        z = w * a
        grad *= float(w * DERIVADAS[ativacao](z))
        a = float(ATIVACOES[ativacao](z))
    return a, grad


# TODO 5b: com x0 = 0,5, imprima |d a_L / d x0| para L em [1, 5, 10, 20, 50] e
#          para sigmoide, tanh, relu e leaky_relu, duas vezes: com w = 1,0 e
#          com w = 3,0 (use  df.to_string(float_format=lambda v: f"{v:.3g}"))
Ls_5b = [1, 5, 10, 20, 50]
ativs_5b = ["sigmoide", "tanh", "relu", "leaky_relu"]
for w_val in [1.0, 3.0]:
    df_5b = pd.DataFrame(
        [{ "L": L, **{at: abs(cadeia(0.5, w_val, L, at)[1]) for at in ativs_5b} } for L in Ls_5b]
    ).set_index("L")
    print(f"\n5b) |d a_L / d x0| com x0 = 0.5 e w = {w_val}:")
    print(df_5b.to_string(float_format=lambda v: f"{v:.3g}"))

# TODO 5c: gere Xu = rng.random((1000, 2)) com rng = np.random.default_rng(SEMENTE).
#          Para o neuronio w = (1; 1) e b em [0, -0,5, -1, -1,5, -2, -3], imprima
#          a fracao de entradas com z > 0 (neuronio ATIVO da ReLU), o gradiente
#          medio da ReLU e o gradiente medio da Leaky ReLU (3 casas)
rng = np.random.default_rng(SEMENTE)
Xu = rng.random((1000, 2))
linhas_5c = []
for b_val in [0.0, -0.5, -1.0, -1.5, -2.0, -3.0]:
    z_u = Neuronio([1.0, 1.0], b_val, "relu").z(Xu)
    linhas_5c.append({
        "b": b_val,
        "frac_ativo (z>0)": np.mean(z_u > 0),
        "grad_medio_relu": np.mean(d_relu(z_u)),
        "grad_medio_leaky": np.mean(d_leaky_relu(z_u)),
    })
df_5c = pd.DataFrame(linhas_5c).set_index("b")
print("\n5c) ReLU morta vs Leaky ReLU:")
print(df_5c.round(3).to_string())


# ============================================================ EXERCICIO 6 (1,5)
secao(6, "O NEURONIO NA IRIS: REGRESSAO LOGISTICA E LINEAR")
iris = load_iris()
mascara = iris.target > 0                       # versicolor (0) x virginica (1)
Xi = StandardScaler().fit_transform(iris.data[mascara][:, [2, 3]])   # comprimento e largura da petala
yi = iris.target[mascara] - 1

# TODO 6a: ajuste LogisticRegression(random_state=SEMENTE) em (Xi, yi). Monte um
#          Neuronio com ativacao "sigmoide", w = lr.coef_[0] e b = lr.intercept_[0].
#          Imprima w, b e a MAIOR diferenca absoluta entre a saida do seu
#          neuronio e lr.predict_proba(Xi)[:, 1]
lr = LogisticRegression(random_state=SEMENTE).fit(Xi, yi)
w_lr, b_lr = lr.coef_[0], lr.intercept_[0]
neu_sig = Neuronio(w_lr, b_lr, "sigmoide")
prob_neu = neu_sig.saida(Xi)
prob_lr = lr.predict_proba(Xi)[:, 1]
diff_6a = np.max(np.abs(prob_neu - prob_lr))
print(f"6a) w = {np.round(w_lr, 4)}, b = {b_lr:.4f}")
print(f"    max |saida_neuronio - predict_proba| = {diff_6a:.2e}")

# TODO 6b: com esses MESMOS w e b, classifique com tres neuronios: sigmoide
#          (classe 1 se y >= 0,5), degrau (y == 1) e tanh (classe 1 se y >= 0).
#          Imprima a acuracia de cada um e em quantas amostras discordam da
#          sigmoide
pred_sig = (neu_sig.saida(Xi) >= 0.5).astype(int)
pred_deg = (Neuronio(w_lr, b_lr, "degrau").saida(Xi) == 1.0).astype(int)
pred_tanh = (Neuronio(w_lr, b_lr, "tanh").saida(Xi) >= 0.0).astype(int)
print("\n6b) Comparacao de classificacao com mesmos w e b:")
for nome, pred in [("sigmoide", pred_sig), ("degrau", pred_deg), ("tanh", pred_tanh)]:
    acc = np.mean(pred == yi)
    disc = int(np.sum(pred != pred_sig))
    print(f"  {nome:<9} acuracia = {acc:.3f} | discordancias da sigmoide = {disc}")

# TODO 6c: ajuste LinearRegression() em (Xi, yi) e monte um neuronio de ativacao
#          "identidade" com esses coeficientes. Imprima a acuracia (classe 1 se
#          y >= 0,5), quantas saidas ficam FORA de [0, 1] (e o minimo e o
#          maximo) e quantas saidas da sigmoide ficam fora de [0, 1]
lin_reg = LinearRegression().fit(Xi, yi)
neu_id = Neuronio(lin_reg.coef_, lin_reg.intercept_, "identidade")
y_id = neu_id.saida(Xi)
acc_id = np.mean((y_id >= 0.5).astype(int) == yi)
fora_id = int(np.sum((y_id < 0.0) | (y_id > 1.0)))
fora_sig = int(np.sum((prob_neu < 0.0) | (prob_neu > 1.0)))
print(f"\n6c) Regressao Linear (neuronio identidade):")
print(f"  acuracia = {acc_id:.3f} | fora de [0, 1] = {fora_id} (min={y_id.min():.3f}, max={y_id.max():.3f})")
print(f"  sigmoide fora de [0, 1] = {fora_sig}")


# ============================================================ EXERCICIO 7 (1,0)
secao(7, "SOFTMAX: ESTABILIDADE E TEMPERATURA")

# TODO 7a: escreva softmax_ingenua(z) = exp(z) / soma e softmax(z, T=1.0), que
#          divide z por T e subtrai o maximo antes do exp. Teste as duas em
#          z = (1000, 1001, 1002) (envolva a ingenua em np.errstate(all="ignore"))
#          e a estavel tambem em (1, 2, 3)
def softmax_ingenua(z):
    e = np.exp(np.asarray(z, dtype=float))
    return e / e.sum()


def softmax(z, T=1.0):
    z_t = np.asarray(z, dtype=float) / T
    e = np.exp(z_t - z_t.max())
    return e / e.sum()


z_grande = [1000, 1001, 1002]
with np.errstate(all="ignore"):
    print("7a) softmax_ingenua([1000, 1001, 1002]):", softmax_ingenua(z_grande))
print("    softmax estavel([1000, 1001, 1002]):", np.round(softmax(z_grande), 4))
print("    softmax estavel([1, 2, 3])         :", np.round(softmax([1, 2, 3]), 4))

# TODO 7b: com z0 = (3; 1; 0) e T em [0,5, 1, 2, 10], imprima as probabilidades,
#          a soma e a entropia -sum(p * ln(p)). Imprima tambem ln(3)
z0 = np.array([3.0, 1.0, 0.0])
print("\n7b) Efeito da temperatura T em z0 = [3, 1, 0]:")
for T in [0.5, 1.0, 2.0, 10.0]:
    p = softmax(z0, T=T)
    entropia = -np.sum(p * np.log(p))
    print(f"  T = {T:<4} p = {np.round(p, 4).tolist()} | soma = {p.sum():.4f} | entropia = {entropia:.4f}")
print(f"  ln(3) (entropia maxima uniforme) = {np.log(3):.4f}")

# TODO 7c: confira numericamente que softmax((2; 0,5))[0] e igual a
#          sigmoide(2 - 0,5)
s_7c = softmax([2.0, 0.5])[0]
sig_7c = sigmoide(2.0 - 0.5)
print(f"\n7c) softmax([2, 0.5])[0] = {s_7c:.6f} | sigmoide(2 - 0.5) = {sig_7c:.6f} | iguais = {np.isclose(s_7c, sig_7c)}")


# ============================================================ RELATORIO
"""
=======================================================================
RELATORIO  (vale 60% da nota desta sessao)
=======================================================================

Ex.1 - (a) Qual e a unica diferenca de codigo entre calcular a saida de
       UM vetor e de um LOTE? (b) Por que o mesmo z produz saidas tao
       diferentes conforme a ativacao? Use os numeros da sua tabela.
    R: (a) No codigo do neuronio nao muda nada, porque o `X @ self.w + self.b`
       ja serve tanto pra um vetor quanto pra matriz do lote inteiro. A unica
       diferenca e se passamos uma amostra por vez num laco `for` ou a matriz
       toda de uma vez.
       (b) Porque cada funcao transforma o `z` para uma faixa diferente. Por
       exemplo, para z = -1: o degrau zera (0.0), a identidade mantem -1.0,
       a sigmoide joga para 0.269 (entre 0 e 1), a tanh da -0.762 (entre -1
       e 1), a ReLU corta para 0.0 e a Leaky ReLU deixa passar -0.01.

Ex.2 - (a) Quais ativacoes tem media proxima de zero em [-10, 10] e por
       que isso importa? (b) Qual fracao de pontos fica saturada em cada
       ativacao e o que isso significa para o treinamento? (c) Por que o
       degrau nao serve para treinar por gradiente?
    R: (a) So a tanh teve media 0.000 (degrau e sigmoide deram 0.500, ReLU
       2.501 e Leaky ReLU 2.476). Isso importa porque saidas centradas em
       zero nao empurram as entradas da proxima camada sempre pro mesmo sinal,
       ajudando o treino a convergir mais rapido.
       (b) A fracao saturada foi: degrau 1.000, tanh 0.701, sigmoide 0.542,
       ReLU 0.500 e Leaky ReLU 0.000. Quando satura, a derivada fica quase
       zero e os pesos praticamente nao atualizam no treinamento.
       (c) Porque a derivada do degrau e sempre zero, entao o gradiente zera
       e nao tem como saber para qual lado ajustar os pesos.

Ex.3 - (a) Por que AND, OR, NOT e MAJ3 tem solucao e XOR/XNOR nao tem
       nenhuma na grade? (b) O que a multiplicacao dos pesos por k = 10
       fez com a saida sigmoide? Em que situacao ela se aproxima do degrau?
    R: (a) Um neuronio so consegue passar uma reta (ou plano) para separar
       as classes. AND (10 solucoes), OR (30), NOT e MAJ3 sao linearmente
       separaveis, mas XOR e XNOR nao sao (deram 0 solucoes).
       (b) Com k = 10, as saidas que eram [0.1824, 0.3775, 0.3775, 0.6225]
       foram para [0.0, 0.0067, 0.0067, 0.9933], bem mais perto de 0 e 1.
       Quanto maior o fator k (pesos grandes), mais inclinada fica a curva
       da sigmoide e mais parecida com o degrau ela fica.

Ex.4 - (a) Por que a rede de 4b devolve a mesma saida para as quatro
       entradas? (b) O que a camada oculta do 4a "faz" com os pontos para
       que o neuronio de saida consiga separa-los? (c) Por que a ReLU na
       oculta resolve o XOR e a identidade nao?
    R: (a) Usando identidade, as duas camadas viram uma conta linear so.
       Como os pesos de OR [1, 1] e NAND [-1, -1] se somam e se anulam
       (W' = [[0, 0]]), a entrada some e sobra so o vies b' = -0.5.
       (b) Ela joga os pontos (0,1) e (1,0) para a mesma posicao (1,1),
       deixando (0,0) em (0,1) e (1,1) em (1,0). Ai uma simples porta AND
       na saida consegue separar.
       (c) Porque a ReLU e nao linear (zera a parte negativa), enquanto
       empilhar camadas com identidade continua sendo apenas uma funcao
       linear igual a um neuronio so.

Ex.5 - (a) Descreva o que acontece com o gradiente da sigmoide e da tanh
       quando L cresce, com w = 1 e com w = 3. (b) A ReLU resolve o
       problema? O que acontece com ela quando w = 3? (c) O que a tabela
       do item 5c mostra sobre a ReLU morta e como a Leaky ReLU ajuda?
    R: (a) Na sigmoide o gradiente some nos dois casos (vai a ~10^-33 com
       w=1 e ~10^-40 com w=3 em L=50). Na tanh com w=1 ele cai mais devagar
       (0.034 em L=50), mas com w=3 a ativacao satura e o gradiente zera
       mais rapido ainda (5.2e-75 em L=50).
       (b) Com w=1 a ReLU mantem o gradiente em 1, mas com w=3 ele multiplica
       por 3 a cada camada e explode (7.18e+23 em L=50).
       (c) Quando o vies b chega em -2 ou -3, todas as entradas ficam com
       z <= 0 e a ReLU morre (gradiente medio 0.000). A Leaky ReLU resolve
       isso porque deixa uma derivada de 0.01 no lado negativo.

Ex.6 - (a) O que a diferenca ~0 do item 6a mostra sobre regressao logistica
       e neuronio? (b) Se a fronteira e a mesma nas tres ativacoes, para
       que serve a sigmoide? (c) Qual problema aparece nas saidas do
       neuronio identidade quando a resposta e uma probabilidade?
    R: (a) Mostra que a regressao logistica nada mais e do que um neuronio
       com ativacao sigmoide (a diferenca deu 0).
       (b) Na hora de classificar, z >= 0 da no mesmo nas tres (acuracia de
       0.940 e 0 discordancias). A vantagem da sigmoide e ter derivada para
       treinar os pesos e dar o resultado entre 0 e 1 como probabilidade.
       (c) Na identidade, 28 amostras passaram de [0, 1] (deu de -0.266 ate
       1.308), o que nao faz sentido como probabilidade.

Ex.7 - (a) Por que a softmax ingenua falha e a estavel nao? (b) Como a
       temperatura muda a distribuicao e a entropia? (c) O que o item 7c
       diz sobre a sigmoide e a softmax?
    R: (a) Na ingenua, `exp(1000)` estoura o limite do float (da `inf`) e a
       divisao vira `nan`. Subtraindo o maximo antes do `exp`, o maior valor
       vira 0 e nao da overflow, mantendo o resultado [0.09, 0.2447, 0.6652].
       (b) T menor (0.5) concentra quase tudo na maior classe (0.9796) e
       baixa a entropia (0.1069). T maior (10) deixa as probabilidades mais
       parecidas e sobe a entropia para 1.0906, perto do maximo ln(3)=1.0986.
       (c) Mostra que para 2 classes a softmax da o mesmo valor que a
       sigmoide da diferenca `z1 - z2` (0.817574), ou seja, a softmax e a
       sigmoide generalizada para mais classes.

=======================================================================
"""

print("""
=======================================================================
Arquivo executado. Confira se todos os TODO foram preenchidos e se o
bloco RELATORIO esta respondido antes de entregar.
=======================================================================""")
