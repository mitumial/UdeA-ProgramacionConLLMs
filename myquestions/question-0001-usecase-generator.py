# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split

def generar_caso_de_uso_procesar_reseñas():
	"""
	Genera un caso de uso aleatorio para la función procesar_reseñas.
	Devuelve:
		input: dict con claves y valores para los argumentos de procesar_reseñas
		output: lo que se esperaría que la función produjera con ese input (simulado)
	"""
	# Parámetros aleatorios
	test_size = random.choice([0.2, 0.25, 0.3])
	max_features = random.choice([10, 20, 50, 100])

	# Generar un DataFrame de ejemplo
	n = random.randint(10, 30)
	textos = [
		random.choice([
			"Me encantó el producto, llegó rápido y funciona perfecto.",
			"No me gustó, mala calidad y llegó tarde.",
			"Excelente servicio y calidad.",
			"No lo recomiendo, muy malo.",
			"Producto aceptable por el precio.",
			"La mejor compra que he hecho.",
			"No cumple con lo prometido.",
			"Muy satisfecho con la compra.",
			"No volvería a comprar.",
			"Superó mis expectativas."
		]) for _ in range(n)
	]
	word_count = [len(t.split()) for t in textos]
	# ratings con algunos nulos
	ratings = [random.choice([1, 2, 3, 4, 5, None]) for _ in range(n)]
	# sentiments con algunos nulos
	sentiments = [random.choice(["positivo", "negativo", None]) for _ in range(n)]

	df = pd.DataFrame({
		"review_text": textos,
		"word_count": word_count,
		"rating": ratings,
		"sentiment": sentiments
	})

	# Input dict
	input_dict = {
		"df": df,
		"test_size": test_size,
		"max_features": max_features
	}

	# Simulación de output esperado (estructura, no valores reales)
	# Eliminar nulos en sentiment
	df_clean = df.dropna(subset=["sentiment"]).copy()
	# Imputar rating con la mediana
	mediana = df_clean["rating"].median()
	df_clean["rating"] = df_clean["rating"].fillna(mediana)
	# Simular vectorización (matriz de tamaño [len(df_clean), max_features])
	X_text = np.random.rand(len(df_clean), max_features)
	# Concatenar word_count y rating
	X_num = df_clean[["word_count", "rating"]].to_numpy()
	X = np.concatenate([X_text, X_num], axis=1)
	y = df_clean["sentiment"].to_numpy()
	# Simular split estratificado
	
	X_train, X_test, y_train, y_test = train_test_split(
		X, y, test_size=test_size, stratify=y, random_state=42
	)
	# Simular vectorizer y modelo
	vectorizer = f"TfidfVectorizer(max_features={max_features})"
	svm_model = "SVC(kernel='linear') (simulado)"

	output = (X_train, X_test, y_train, y_test, vectorizer, svm_model)
	return input_dict, output
