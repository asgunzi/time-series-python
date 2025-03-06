# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 19:20:59 2025

@author: asgun
"""

import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt

# Parâmetros do áudio
sample_rate = 44100  # Taxa de amostragem (Hz)

# Exemplo de vetor
#vetor = np.sin(2 * np.pi * 440 * np.linspace(0, 1, sample_rate))  
# Tom de 440 Hz por 1 segundo
# Trocar frequência para ouvir o que acontece

# Ruído branco
vetor = np.random.rand(sample_rate)

#Média movel
v2 =[]
janela=60
for idx in range(len(vetor)-janela):
    v2.append(np.mean(vetor[idx:idx+janela]))
vetor = v2

# Passa alta
# v2 =[]
# for idx in range(len(vetor)-1):
#     v2.append(2*vetor[idx]-vetor[idx+1])
# vetor = v2


def reproduzir_audio(vetor, sample_rate):
    sd.play(vetor, samplerate=sample_rate)
    sd.wait()  # Aguarda a reprodução terminar

# Reproduzindo o vetor
reproduzir_audio(vetor, sample_rate)

#Plotando parte do vetor
plt.plot(vetor[:500])
plt.show()

