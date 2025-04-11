#!/usr/bin/env python3
"""
6-baum_welch.py
"""


import numpy as np


def backward(Observations, Emission, Transition, Initial):
    """backward"""
    T = Observations.shape[0]
    N = Emission.shape[0]
    B = np.zeros((N, T))
    B[:, T-1] = 1
    for t in range(T-2, -1, -1):
        for i in range(N):
            B[i, t] = np.sum(
                B[:, t + 1] * Transition[i, :] * Emission[
                    :, Observations[t+1]])
    P = np.sum(Initial.T * Emission[:, Observations[0]] * B[:, 0])

    return P, B


def forward(Observations, Emission, Transition, Initial):
    """"foward"""
    T = Observations.shape[0]
    N = Emission.shape[0]
    F = np.zeros((N, T))

    F[:, 0] = Initial[:, 0] * Emission[:, Observations[0]]
    for t in range(1, T):
        for i in range(N):
            F[i, t] = np.sum(
                F[:, t - 1] * Transition[:, i]) * Emission[i, Observations[t]]

    P = np.sum(F[:, T - 1])
    return P, F


def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):
    """baum_welch"""
    M, N = Emission.shape
    T = Observations.shape[0]

    for n in range(iterations):
        P_f, F = forward(
            Observations, Emission, Transition, Initial)
        P_b, B = backward(
            Observations, Emission, Transition, Initial)
        G = np.zeros((M, T))
        xi = np.zeros((M, M, T-1))
        for t in range(T):
            for i in range(M):
                G[i, t] = (F[i, t] * B[i, t]) / np.sum(F[:, t] * B[:, t])

        for t in range(T - 1):
            for i in range(M):
                for j in range(M):
                    xi[i, j, t] = (
                        F[i, t] * Transition[i, j] * Emission[
                            j, Observations[t + 1]] * B[
                                j, t + 1]) / np.sum(F[:, t] * B[:, t])

        for i in range(M):
            for j in range(M):
                Transition[i, j] = np.sum(
                    xi[i, j, :]) / np.sum(G[i, :T - 1])

        for j in range(M):
            for k in range(N):
                Emission[j, k] = np.sum(
                    G[j, :] * (Observations == k)) / np.sum(G[j, :])

    return Transition, Emission
