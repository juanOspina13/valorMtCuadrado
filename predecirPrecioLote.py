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
costo_metro_cuadrado = 5

#  Generamos precios de casas añadiendo un ruido aleatorio desde 1000mil hasta 8 Millones (comision )
#np.random.seed(42)
precios_lotes = tamanos_lotes * costo_metro_cuadrado
# + np.random.randint(low=100000, high=8000000, size=numero_casas)  

def normalizar(array):
    return (array - array.mean()) / array.std()

def desnormalizar(array):
    return (array + array.mean()) * array.std()

def desnormalizarResultado(valor,array):
    return (valor + (array.mean())/ array.std())

# define training data
num_train_samples = math.floor(numero_casas * 0.7)

tamano_lote_entreno = np.asarray(tamanos_lotes[:num_train_samples])
precios_lote_entreno = np.asarray(precios_lotes[:num_train_samples:])

tamano_lote_entreno_normalizado = normalizar(tamano_lote_entreno)
precio_lote_entreno_normalizado = normalizar(precios_lote_entreno)



tamano_lote_pruebas = np.array(tamanos_lotes[num_train_samples:])
precios_lote_pruebas = np.array(precios_lotes[num_train_samples:])

tamano_lote_pruebas_normalizado = normalizar(tamano_lote_pruebas)
precios_lote_pruebas_normalizado = normalizar(precios_lote_pruebas)



# Pintamos la grafica de precios y tamaño
plt.plot(tamanos_lotes, precios_lotes, "bx")
plt.ylabel("Precio")
plt.xlabel("Metros Cuadrados")
plt.show()

numero_datos_entreno = 10
tf_precio_mt_cuadrado = tf.Variable(3000.,name="tf_precio_mt_cuadrado")

tf_valor_lote = tf.placeholder("float", name="tf_valor_lote")
tf_tamanos_lote = tf.placeholder("float", name="tf_tamano_lote")

tf_precio_mt_cuadrado_pred = tf.multiply(tf_precio_mt_cuadrado,  tf_tamanos_lote)

# 3. Funcion del costo de error cuadratico
tf_costo = tf.reduce_sum(tf.pow(tf_precio_mt_cuadrado_pred-tf_valor_lote, 2))/(2*10)

# 4. Vamos a utilizar el algoritmo de gradiente descendiente
learning_rate = 0.1
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(tf_costo)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    iteraciones_aprendizaje = 200

    for iteration in range(iteraciones_aprendizaje):
        for(x,y) in zip(tamano_lote_entreno_normalizado, precio_lote_entreno_normalizado):
            optimo = sess.run(optimizer, feed_dict={tf_tamanos_lote:x,tf_valor_lote:y})
        
        costo = sess.run(tf_costo, feed_dict={tf_tamanos_lote:tamano_lote_entreno_normalizado,tf_valor_lote:precio_lote_entreno_normalizado})
        #hipotesis = sess.run(tf_precio_mt_cuadrado_pred, feed_dict={tf_tamano_lote:x})
        #print("Tamano Lote=", "{:.9f}".format(x))            
        #print("Hipotesis=", "{:.9f}".format(hipotesis))
        #print("Costo=", "{:.9f}".format(costo))
    #print("Valor Mt Cuadrado =>",sess.run(tf_precio_mt_cuadrado))
 
        
            

    
    # get values used to normalized data so we can denormalize data back to its original scale
    media_tamano_lote_entreno = tamano_lote_entreno.mean()
    std_tamano_lote_entreno = tamano_lote_entreno.std()

    media_precios_lote_entreno = precios_lote_entreno.mean()
    std_precios_lote_entreno = precios_lote_entreno.std()

    std_precios_lote = precios_lotes.std()
    media_precios_lote = precios_lotes.mean()

    # Plot the graph
    plt.rcParams["figure.figsize"] = (10,8)
    plt.figure()
    plt.ylabel("Price")
    plt.xlabel("Size (sq.ft)")
    plt.plot(tamano_lote_entreno, precios_lote_entreno, 'go', label='Informacion Entreno')
    plt.plot(tamano_lote_pruebas, precios_lote_pruebas, 'mo', label='Informacion pruebas')
    plt.plot(
            tamano_lote_entreno_normalizado *std_tamano_lote_entreno + media_tamano_lote_entreno,
             ((sess.run(tf_precio_mt_cuadrado) * tamano_lote_entreno_normalizado ))*std_precios_lote_entreno+media_precios_lote_entreno,
                 label='Regresion Aprendida')
             
    plt.legend(loc='upper left')
    plt.show()
    #print(precio_lote_entreno_normalizado)
    #print(precios_lotes)    
    #print("Valor Mt Cuadrado =>",format(sess.run(tf_precio_mt_cuadrado)* std_precios_lote_entreno+ media_precios_lote_entreno))
    #print("***Valor en 200")
    #print(
        #(sess.run(tf_precio_mt_cuadrado) * normalizar( np.array( [200, 100] ) ) * std_precios_lote_entreno + media_precios_lote_entreno ) )