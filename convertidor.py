import tensorflow as tf 

# Carga tu modelo entrenado
model = tf.keras.models.load_model('./modelo/modelo.h5')

# Convierte a TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Guarda el modelo TFLite
with open('modelo_pies.tflite', 'wb') as f:
    f.write(tflite_model)