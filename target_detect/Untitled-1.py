# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'
#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'target_detect'))
	print(os.getcwd())
except:
	pass
#%% [markdown]
# Universidade Federal de Minas Gerais
# # Targets
# 
# Arthur Phillip Ferreira da Silva
# Gabriel Almeida de Jesus
#%% [markdown]
# Bibliotecas utilizadas

#%%
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

#%% [markdown]
# Importando imagem base em escala de cinza

#%%
image = cv.imread('cena1.png',0)

plt.imshow(image, cmap='gray')

#%% [markdown]
# Fazendo a binarização da imagem

#%%
(T, binary) = cv.threshold(image, 127, 255, cv.THRESH_BINARY_INV)

plt.subplot(121),plt.imshow(image,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(binary,cmap = 'gray')
plt.title('Binary Image'), plt.xticks([]), plt.yticks([])

plt.show()

#%% [markdown]
# Detectando bordas

#%%
edges = cv.Canny(binary,100,200,3)

plt.subplot(121),plt.imshow(image,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

plt.show()

#%% [markdown]
# Detectando quinas

#%%
corners = cv.cornerHarris(binary,2,3,0.04)

plt.subplot(121),plt.imshow(image,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(corners,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

plt.show()


#%%
suave = cv.GaussianBlur(image, (7, 7), 0)

plt.subplot(121),plt.imshow(image,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(suave,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

plt.show()


#%%
canny1 = cv.Canny(suave, 20, 120)
canny2 = cv.Canny(suave, 70, 200)

plt.subplot(121),plt.imshow(canny1,cmap = 'gray')
plt.title('Teste 1'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(canny2,cmap = 'gray')
plt.title('Teste 2'), plt.xticks([]), plt.yticks([])

plt.show()

#%%

