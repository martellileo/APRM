from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import confusion_matrix


BASE_DIR = Path(__file__).resolve().parent


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
	estufas = pd.read_csv(BASE_DIR / "estufas_energia.csv")
	lotes = pd.read_csv(BASE_DIR / "lotes_cafe.csv")
	return estufas, lotes


def regression_pipeline(degree: int) -> Pipeline:
	return Pipeline(
		[
			("scaler", StandardScaler()),
			("poly", PolynomialFeatures(degree=degree, include_bias=False)),
			("model", LinearRegression()),
		]
	)


def knn_pipeline(k: int) -> Pipeline:
	return Pipeline(
		[
			("scaler", StandardScaler()),
			("model", KNeighborsClassifier(n_neighbors=k)),
		]
	)


def evaluate_regression(estufas: pd.DataFrame) -> pd.DataFrame:
	x = estufas[["temperatura_externa_c", "umidade_relativa_pct"]]
	y = estufas["consumo_kwh_dia"]
	cv = KFold(n_splits=5, shuffle=True, random_state=42)

	rows = []
	for degree in range(1, 9):
		pipeline = regression_pipeline(degree)
		scores = cross_validate(
			pipeline,
			x,
			y,
			cv=cv,
			scoring={
				"rmse": "neg_root_mean_squared_error",
				"mae": "neg_mean_absolute_error",
				"r2": "r2",
			},
			return_train_score=True,
		)
		rows.append(
			{
				"grau": degree,
				"rmse_treino_medio": (-scores["train_rmse"]).mean(),
				"rmse_validacao_medio": (-scores["test_rmse"]).mean(),
			}
		)

	return pd.DataFrame(rows)


def regression_fold_metrics(estufas: pd.DataFrame, degree: int) -> pd.DataFrame:
	x = estufas[["temperatura_externa_c", "umidade_relativa_pct"]]
	y = estufas["consumo_kwh_dia"]
	cv = KFold(n_splits=5, shuffle=True, random_state=42)

	pipeline = regression_pipeline(degree)
	scores = cross_validate(
		pipeline,
		x,
		y,
		cv=cv,
		scoring={
			"rmse": "neg_root_mean_squared_error",
			"mae": "neg_mean_absolute_error",
			"r2": "r2",
		},
	)

	return pd.DataFrame(
		{
			"fold": range(1, 6),
			"rmse": -scores["test_rmse"],
			"mae": -scores["test_mae"],
			"r2": scores["test_r2"],
		}
	)


def fit_regression_final(estufas: pd.DataFrame, degree: int) -> Pipeline:
	x = estufas[["temperatura_externa_c", "umidade_relativa_pct"]]
	y = estufas["consumo_kwh_dia"]
	pipeline = regression_pipeline(degree)
	pipeline.fit(x, y)
	return pipeline


def evaluate_knn(lotes: pd.DataFrame) -> pd.DataFrame:
	x = lotes[["altitude_m", "umidade_grao_pct"]]
	y = lotes["classe_bebida"]
	cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

	rows = []
	for k in range(1, 26, 2):
		pipeline = knn_pipeline(k)
		scores = cross_validate(
			pipeline,
			x,
			y,
			cv=cv,
			scoring={"acc": "accuracy", "f1": "f1_macro"},
			return_train_score=True,
		)
		rows.append(
			{
				"k": k,
				"acc_treino_medio": scores["train_acc"].mean(),
				"acc_validacao_medio": scores["test_acc"].mean(),
				"f1_treino_medio": scores["train_f1"].mean(),
				"f1_validacao_medio": scores["test_f1"].mean(),
			}
		)

	return pd.DataFrame(rows)


def knn_fold_metrics(lotes: pd.DataFrame, k: int) -> tuple[pd.DataFrame, pd.DataFrame]:
	x = lotes[["altitude_m", "umidade_grao_pct"]]
	y = lotes["classe_bebida"]
	cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

	pipeline = knn_pipeline(k)
	scores = cross_validate(
		pipeline,
		x,
		y,
		cv=cv,
		scoring={"acc": "accuracy", "f1": "f1_macro"},
	)

	folds = pd.DataFrame(
		{
			"fold": range(1, 6),
			"acc": scores["test_acc"],
			"f1_macro": scores["test_f1"],
		}
	)

	predicted = cross_val_predict(pipeline, x, y, cv=cv)
	labels = sorted(y.unique())
	matrix = pd.DataFrame(
		confusion_matrix(y, predicted, labels=labels),
		index=[f"real_{label}" for label in labels],
		columns=[f"pred_{label}" for label in labels],
	)

	return folds, matrix


def fit_knn_final(lotes: pd.DataFrame, k: int) -> Pipeline:
	x = lotes[["altitude_m", "umidade_grao_pct"]]
	y = lotes["classe_bebida"]
	pipeline = knn_pipeline(k)
	pipeline.fit(x, y)
	return pipeline


def main() -> None:
	estufas, lotes = load_data()

	regression_summary = evaluate_regression(estufas)
	best_degree = int(
		regression_summary.sort_values(
			["rmse_validacao_medio", "grau"], ascending=[True, True]
		).iloc[0]["grau"]
	)

	regression_folds = regression_fold_metrics(estufas, best_degree)
	regression_model = fit_regression_final(estufas, best_degree)
	regression_new = pd.DataFrame(
		{
			"temperatura_externa_c": [8, 21, 35],
			"umidade_relativa_pct": [70, 55, 85],
		}
	)
	regression_predictions = regression_model.predict(regression_new)

	knn_summary = evaluate_knn(lotes)
	best_k = int(
		knn_summary.sort_values(
			["acc_validacao_medio", "k"], ascending=[False, True]
		).iloc[0]["k"]
	)

	knn_folds, knn_matrix = knn_fold_metrics(lotes, best_k)
	knn_model = fit_knn_final(lotes, best_k)
	knn_new = pd.DataFrame(
		{
			"altitude_m": [1000, 1350, 1250],
			"umidade_grao_pct": [12.0, 12.4, 11.0],
		}
	)
	knn_predictions = knn_model.predict(knn_new)

	print("Exercicio 1 - regressao polinomial")
	print(regression_summary.to_string(index=False))
	print("\nGrau escolhido:", best_degree)
	print(regression_folds.to_string(index=False))
	print("Predicoes finais:")
	for label, value in zip(["A", "B", "C"], regression_predictions):
		print(f"  {label}: {value:.2f} kWh/dia")

	print("\nExercicio 2 - K-NN")
	print(knn_summary.to_string(index=False))
	print("\nK escolhido:", best_k)
	print(knn_folds.to_string(index=False))
	print("\nMatriz de confusao:")
	print(knn_matrix.to_string())
	print("\nPredicoes finais:")
	for label, value in zip(["X", "Y", "Z"], knn_predictions):
		print(f"  {label}: {value}")


if __name__ == "__main__":
	main()
