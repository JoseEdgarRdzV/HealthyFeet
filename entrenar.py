# Librerías para manejar el sistema operativo
import os

# Preprocesamiento de imágenes
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Optimizador para entrenamiento
from tensorflow.keras import optimizers

# Creación de modelos secuenciales
from tensorflow.keras.models import Sequential

# Capas de la red neuronal
from tensorflow.keras.layers import Dropout, Flatten, Dense, Convolution2D, MaxPooling2D

# Para limpiar sesiones previas
from tensorflow.keras import backend as K

# Limpiar la sesión previa
K.clear_session()

# Directorios de datos
data_entrenamiento = "./data/entrenamiento"
data_validacion = "./data/validacion"

# Hiperparámetros
epocas = 10
altura, longitud = 100, 100
batch_size = 32
pasos = 500
pasos_validacion = 200
filtrosConv1 = 32
filtrosConv2 = 64
tamano_filtro1 = (3, 3)
tamano_filtro2 = (2, 2)
tamano_pool = (2, 2)
clases = 3
lr = 0.0005

# Preprocesamiento de imágenes
entrenamiento_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True
)

validacion_datagen = ImageDataGenerator(
    rescale=1. / 255
)

# Carga de imágenes de entrenamiento
imagen_entrenamiento = entrenamiento_datagen.flow_from_directory(
    data_entrenamiento,
    target_size=(altura, longitud),
    batch_size=batch_size,
    class_mode='categorical'
)

# Carga de imágenes de validación
imagen_validacion = validacion_datagen.flow_from_directory(
    data_validacion,
    target_size=(altura, longitud),
    batch_size=batch_size,
    class_mode='categorical'
)


# Creación de la CNN
cnn = Sequential()

# Primera capa convolucional
cnn.add(Convolution2D(filtrosConv1, tamano_filtro1, padding='same', input_shape=(altura, longitud, 3), activation='relu'))
cnn.add(MaxPooling2D(pool_size=tamano_pool))

# Segunda capa convolucional
cnn.add(Convolution2D(filtrosConv2, tamano_filtro2, padding='same', activation='relu'))
cnn.add(MaxPooling2D(pool_size=tamano_pool))

# Aplanamiento de la red
cnn.add(Flatten())
cnn.add(Dense(256, activation='relu'))
cnn.add(Dropout(0.5))  # Evita el sobreajuste

# Capa de salida
cnn.add(Dense(clases, activation='softmax'))

# Compilación del modelo
cnn.compile(
    loss='categorical_crossentropy',
    optimizer=optimizers.Adam(learning_rate=lr),
    metrics=['accuracy']
)

# Entrenamiento del modelo
cnn.fit(
    imagen_entrenamiento,
    steps_per_epoch=pasos,
    epochs=epocas,
    validation_data=imagen_validacion,
    validation_steps=pasos_validacion
)

# Guardar el modelo entrenado
dir = './modelo/'

if not os.path.exists(dir):
    os.mkdir(dir)

cnn.save('./modelo/modelo.h5')
cnn.save_weights('./modelo/pesos.h5')

