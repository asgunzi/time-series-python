# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 08:25:09 2025

@author: asgun
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Carregar os dados em um DataFrame do Pandas
data = np.array([0.33, -0.31, 0.02, 0.36, 2.71, 2.03, 2.39, 2.71, 3.06, 3.40, 2.73, 3.09, 4.41, 4.77, 5.09, 5.44, 4.78, 7.11, 7.47, 6.79, 7.15, 6.48, 7.82])
# Ajustar o modelo de Holt-Winters
modelo_hw = ExponentialSmoothing(data, seasonal='add', seasonal_periods=6).fit()

# Fazer previsões
previsoes = modelo_hw.forecast(steps=12)

# Exibir as previsões
print(previsoes)


fittedvalues = modelo_hw.fittedvalues
trend = modelo_hw.level
seasonal = modelo_hw.season
resid = data - fittedvalues


# Plotar os componentes
plt.figure(figsize=(12, 8))

# Dados originais e valores ajustados
plt.subplot(4, 1, 1)
plt.plot(data, label='Dados Originais')
plt.plot(fittedvalues, label='Valores Ajustados', color='red')
plt.legend(loc='upper left')
plt.title('Dados Originais e Valores Ajustados')

# Tendência
plt.subplot(4, 1, 2)
plt.plot(trend, label='Tendência', color='green')
plt.legend(loc='upper left')
plt.title('Tendência')

# Sazonalidade
plt.subplot(4, 1, 3)
plt.plot(seasonal, label='Sazonalidade', color='orange')
plt.legend(loc='upper left')
plt.title('Sazonalidade')

# Resíduos
plt.subplot(4, 1, 4)
plt.plot(resid, label='Resíduos', color='purple')
plt.legend(loc='upper left')
plt.title('Resíduos')

plt.tight_layout()
plt.show()