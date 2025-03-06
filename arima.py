# -*- coding: utf-8 -*-
"""
Created on Wed Mar  5 00:53:15 2025

@author: asgun
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Gerando uma série temporal de exemplo

dados = np.array([0.33, -0.31, 0.02, 0.36, 2.71, 2.03, 2.39, 2.71, 3.06, 3.40, 2.73, 3.09, 4.41, 4.77, 5.09, 5.44, 4.78, 7.11, 7.47, 6.79, 7.15, 6.48, 7.82])

serie_temporal = pd.Series(dados)

# Dividindo os dados em treino e teste
tamanho_treino = int(len(serie_temporal) * 0.95)
treino, teste = serie_temporal[:tamanho_treino], serie_temporal[tamanho_treino:]

# Ajustando o modelo ARIMA nos dados de treino
modelo = ARIMA(treino, order=(5, 1, 0))
modelo_ajustado = modelo.fit()

# Fazendo previsões nos dados de teste
previsoes = modelo_ajustado.forecast(steps=len(teste))
previsoes = pd.Series(previsoes, index=teste.index)

# Plotando os resultados
plt.figure(figsize=(10, 6))
plt.plot(treino, label='Treino')
plt.plot(teste, label='Teste')
plt.plot(previsoes, label='Previsões', linestyle='--')
plt.legend()
plt.show()

# Exibindo o resumo do modelo ajustado
print(modelo_ajustado.summary())


