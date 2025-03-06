# -*- coding: utf-8 -*-
"""
Created on Wed Mar  5 22:57:40 2025

@author: asgun
"""

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

dados = pd.read_excel(r"C:\Prof Arnaldo\IBMEC\Time Series\co2.xlsx", skiprows = 40)

serie1 = dados['average']
serie1 =serie1.dropna()

#Plota série original
plt.plot(serie1)
plt.show()

print(serie1)


#Decompõe segundo Holt Winters
modelo_hw = ExponentialSmoothing(serie1, seasonal='add', seasonal_periods=12).fit()

# Fazer previsões
previsoes = modelo_hw.forecast(steps=12)

# Exibir as previsões
print(previsoes)


fittedvalues = modelo_hw.fittedvalues
trend = modelo_hw.level
seasonal = modelo_hw.season
resid = serie1 - fittedvalues


# Plotar os componentes
plt.figure(figsize=(12, 8))

# Dados originais e valores ajustados
plt.subplot(4, 1, 1)
plt.plot(serie1, label='Dados Originais')
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





