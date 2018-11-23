#   Este programa predice el precio que costara un lote dependiendo de su tamano (mts cuadrados)

import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation 

#  generamos lotes entre 100 y 1500 metros cuadrados
numero_casas = 10
np.random.seed(42)
tamanos_lotes = np.random.randint(low=10, high=100, size=numero_casas )
costo_metro_cuadrado = 3

#  Generamos precios de casas añadiendo un ruido aleatorio desde 1000mil hasta 8 Millones (comision )
#np.random.seed(42)
precios_lotes = tamanos_lotes * costo_metro_cuadrado
print(tamanos_lotes)
print(precios_lotes)
# + np.random.randint(low=100000, high=8000000, size=numero_casas)  

def normalizar(array):
    return (array - array.mean()) / array.std()

tamano_lote_normalizado = tamanos_lotes
precios_lotes_normalizado = precios_lotes

# Pintamos la grafica de precios y tamaño
plt.plot(tamano_lote_normalizado, precios_lotes_normalizado, "bx")
plt.ylabel("Precio")
plt.xlabel("Metros Cuadrados")
plt.show()

numero_datos_entreno = 10
tf_precio_mt_cuadrado = tf.Variable( 3.,name="tf_precio_mt_cuadrado")

tf_valor_lote = tf.placeholder("float", name="tf_valor_lote")
tf_tamanos_lote = tf.placeholder("float", name="tf_tamano_lote")

tf_precio_mt_cuadrado_pred = tf.multiply(tf_precio_mt_cuadrado,  tf_tamanos_lote)

# 3. Funcion del costo de error cuadratico
tf_costo = tf.reduce_sum(tf.pow(tf_precio_mt_cuadrado_pred-tf_valor_lote, 2))/(2*10)

# 4. Vamos a utilizar el algoritmo de gradiente descendiente
learning_rate = 0.1
optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(tf_costo)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    iteraciones_aprendizaje = 100

    for iteration in range(iteraciones_aprendizaje):
        for(x,y) in zip(tamano_lote_normalizado, precios_lotes_normalizado):
            optimo = sess.run(optimizer, feed_dict={tf_tamanos_lote:x,tf_valor_lote:y})
        
        costo = sess.run(tf_costo, feed_dict={tf_tamanos_lote:tamano_lote_normalizado,tf_valor_lote:precios_lotes_normalizado})
        #hipotesis = sess.run(tf_precio_mt_cuadrado_pred, feed_dict={tf_tamano_lote:x})
        #print("Tamano Lote=", "{:.9f}".format(x))            
        #print("Hipotesis=", "{:.9f}".format(hipotesis))
        #print("Costo=", "{:.9f}".format(costo))

    print("Valor Mt Cuadrado =>", sess.run(tf_precio_mt_cuadrado))
 
        
            
