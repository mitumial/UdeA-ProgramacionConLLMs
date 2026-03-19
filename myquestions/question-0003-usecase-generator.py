# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import random
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer

def generar_caso_de_uso_analizar_defectos():
	"""
	Genera un caso de uso aleatorio para la función analizar_defectos.
	Devuelve:
		input: dict con claves y valores para los argumentos de analizar_defectos
		output: lo que se esperaría que la función produjera con ese input (simulado)
	"""
	# Parámetros aleatorios
	test_size = random.choice([0.2, 0.25, 0.3])
	pca_components = random.choice([2, 3])

	# Generar un DataFrame de ejemplo
	n = random.randint(30, 60)
	vibration = np.random.normal(loc=5, scale=2, size=n)
	temperature = np.random.normal(loc=50, scale=10, size=n)
	# pressure con algunos valores negativos
	pressure = np.random.normal(loc=100, scale=20, size=n)
	for _ in range(max(1, n//10)):
		idx = random.randint(0, n-1)
		pressure[idx] = -abs(pressure[idx])
	# defect_type multiclase (5 clases) con algunos nulos
	defect_types = [random.choice([0,1,2,3,4,None]) for _ in range(n)]

	df = pd.DataFrame({
		"vibration": vibration,
		"temperature": temperature,
		"pressure": pressure,
		"defect_type": defect_types
	})

	# Input dict
	input_dict = {
		"df": df,
		"test_size": test_size,
		"pca_components": pca_components
	}

	# Simulación de output esperado (estructura, no valores reales)
	# Eliminar nulos en defect_type
	df_clean = df.dropna(subset=["defect_type"]).copy()
	# Reemplazar negativos en pressure
	df_clean["pressure"] = df_clean["pressure"].apply(lambda x: max(x,0))
	# Escalar características
	
	normalizer = Normalizer(norm="l2")
	X_scaled = normalizer.fit_transform(df_clean[["vibration","temperature","pressure"]])
	y = df_clean["defect_type"].to_numpy()
	# PCA
	pca = PCA(n_components=pca_components)
	X_pca = pca.fit_transform(X_scaled)
	# Split estratificado
	X_pca_train, X_pca_test, y_train, y_test = train_test_split(
		X_pca, y, test_size=test_size, stratify=y, random_state=42
	)
	# Simular modelo PCA
	pca_model = f"PCA(n_components={pca_components}) (simulado)"
	output = (X_pca_train, X_pca_test, y_train, y_test, pca_model)
	return input_dict, output
