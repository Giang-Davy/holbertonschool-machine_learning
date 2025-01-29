#!/usr/bin/env python3


import numpy as np
np.allclose(student_output, expected_output, atol=1e-8)


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
	m = Y.shape[1]
	for i in range(L, 0, -1):
		A = cache[f"A{i}"]
		A_prev = cache[f"A{i-1}"] if i > 1 else cache["A0"]
		W = weights[f"W{i}"]
		b = weights[f"b{i}"]
		
		if i == L:
			dZ = A - Y
		else:
			dZ = (1 - A ** 2) * np.dot(weights[f"W{i+1}"].T, dZ)
		
		dW = (np.dot(dZ, A_prev.T) + lambtha * W) / m
		db = np.sum(dZ, axis=1, keepdims=True) / m
		
		weights[f"W{i}"] -= alpha * dW
		weights[f"b{i}"] -= alpha * db
