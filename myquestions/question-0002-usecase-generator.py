# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

def generar_caso_de_uso_preparar_demanda_energia():
	"""
	Genera un caso de uso aleatorio para la función preparar_demanda_energia.
	Devuelve:
		input: dict con claves y valores para los argumentos de preparar_demanda_energia
		output: lo que se esperaría que la función produjera con ese input (simulado)
	"""
	# Parámetros aleatorios
	n_splits = random.choice([3, 4, 5])
	random_seed = random.randint(1, 1000)

	# Generar un DataFrame de ejemplo
	n = random.randint(30, 60)
	base_date = pd.Timestamp('2023-01-01')
	timestamps = [base_date + pd.Timedelta(hours=i) for i in range(n)]
	temperature = np.random.normal(loc=20, scale=5, size=n)
	# Introducir outliers
	for _ in range(max(1, n//20)):
		idx = random.randint(0, n-1)
		temperature[idx] = temperature.mean() + 10 + np.random.rand()*10
	humidity = np.random.uniform(30, 90, size=n)
	holiday = [random.choice([True, False]) for _ in range(n)]
	# energy_demand con algunos nulos
	energy_demand = np.random.uniform(100, 500, size=n)
	for _ in range(max(1, n//15)):
		idx = random.randint(0, n-1)
		energy_demand[idx] = None

	df = pd.DataFrame({
		"timestamp": timestamps,
		"temperature": temperature,
		"humidity": humidity,
		"holiday": holiday,
		"energy_demand": energy_demand
	})

	# Input dict
	input_dict = {
		"df": df,
		"n_splits": n_splits,
		"random_seed": random_seed
	}

	# Simulación de output esperado (estructura, no valores reales)
	# Eliminar nulos en energy_demand
	df_clean = df.dropna(subset=["energy_demand"]).copy()
	# Filtrar outliers en temperature (percentil 99)
	temp_99 = df_clean["temperature"].quantile(0.99)
	df_clean = df_clean[df_clean["temperature"] <= temp_99]
	# Codificar timestamp
	df_clean["hour"] = df_clean["timestamp"].dt.hour
	df_clean["day_of_week"] = df_clean["timestamp"].dt.dayofweek
	# Escalar temperature y humidity
	scaler = StandardScaler()
	scaled = scaler.fit_transform(df_clean[["temperature", "humidity"]])
	X_num = scaled
	# Features de tiempo
	time_features = df_clean[["hour", "day_of_week"]].reset_index(drop=True)
	# Concatenar features
	X = np.concatenate([X_num, time_features.to_numpy(), np.array(df_clean["holiday"]).reshape(-1,1)], axis=1)
	y = df_clean["energy_demand"].to_numpy()
	# Simular TimeSeriesSplit
	tscv = TimeSeriesSplit(n_splits=n_splits)
	# Tomar el último split como test
	splits = list(tscv.split(X))
	train_idx, test_idx = splits[-1]
	X_train, X_test = X[train_idx], X[test_idx]
	y_train, y_test = y[train_idx], y[test_idx]
	# output
	output = (X_train, X_test, y_train, y_test, time_features)
	return input_dict, output
