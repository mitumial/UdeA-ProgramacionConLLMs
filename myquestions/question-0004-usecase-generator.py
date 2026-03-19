# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import StratifiedShuffleSplit

def generar_caso_de_uso_generar_segmentos():
	"""
	Genera un caso de uso aleatorio para la función generar_segmentos.
	Devuelve:
		input: dict con claves y valores para los argumentos de generar_segmentos
		output: lo que se esperaría que la función produjera con ese input (simulado)
	"""
	# Parámetros aleatorios
	test_size = random.choice([0.2, 0.25, 0.3])
	n_clusters = random.choice([2, 3, 4, 5])

	# Generar un DataFrame de ejemplo
	n = random.randint(40, 80)
	age = np.random.randint(18, 70, size=n)
	# income con nulos estratificados por age
	income = np.random.normal(loc=30000, scale=10000, size=n)
	for i in range(n):
		if random.random() < 0.1:  # 10% nulos
			income[i] = None
	purchase_frequency = np.random.randint(1, 20, size=n)
	cltv = np.random.uniform(100, 10000, size=n)
	# cluster con algunos nulos
	cluster = [random.choice([0,1,2,3,None]) for _ in range(n)]

	df = pd.DataFrame({
		"age": age,
		"income": income,
		"purchase_frequency": purchase_frequency,
		"cltv": cltv,
		"cluster": cluster
	})

	# Input dict
	input_dict = {
		"df": df,
		"test_size": test_size,
		"n_clusters": n_clusters
	}

	# Simulación de output esperado (estructura, no valores reales)
	# Eliminar nulos en cluster
	df_clean = df.dropna(subset=["cluster"]).copy()
	# Imputar income con la media por grupo de age (rangos de 10 años)
	df_clean["age_group"] = (df_clean["age"]//10)*10
	df_clean["income"] = df_clean.groupby("age_group")["income"].transform(lambda x: x.fillna(x.mean()))
	# Escalar todas las características
	scaler = RobustScaler()
	X = scaler.fit_transform(df_clean[["age","income","purchase_frequency","cltv"]])
	y = df_clean["cluster"].to_numpy()
	# StratifiedShuffleSplit
	sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
	train_idx, test_idx = next(sss.split(X, y))
	X_scaled_train, X_scaled_test = X[train_idx], X[test_idx]
	y_train, y_test = y[train_idx], y[test_idx]
	# Simular modelo KMeans
	kmeans_model = f"KMeans(n_clusters={n_clusters}) (simulado)"
	output = (X_scaled_train, X_scaled_test, y_train, y_test, kmeans_model)
	return input_dict, output
