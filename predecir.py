import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model

# Dimensiones de las imágenes
longitud, altura = 100, 100

# Carga del modelo
modelo = './modelo/modelo.h5'
cnn = load_model(modelo)

# Función de predicción
def predict(file):
    x = load_img(file, target_size=(longitud, altura))  # Carga la imagen
    x = img_to_array(x)  # Convierte la imagen a un arreglo
    x = x / 255.0  # Normaliza la imagen
    x = np.expand_dims(x, axis=0)  # Agrega una dimensión extra para el batch
    arreglo = cnn.predict(x)  # Realiza la predicción
    resultado = arreglo[0]
    respuesta = np.argmax(resultado)  # Obtiene la clase con mayor probabilidad
    if respuesta == 0:
        print('Cavo')
    elif respuesta == 1:
        print('Plano')
    elif respuesta == 2:
        print('Sano')
    return respuesta

# Descomenta para probar
predict('dataPlano.png')
