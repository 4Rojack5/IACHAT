import nltk
from nltk.stem.lancaster import LancasterStemmer ##Interpretar palabras
stemmer  = LancasterStemmer()
import numpy
import tflearn
import tensorflow
import json
import random
import pickle

nltk.download('punkt')

with open("content.json") as archive:
    data = json.load(archive)

words = []
tags  = []
auxX  = []
auxY  = []

for content in data["content"]:
    for patterns in content["patterns"]:
        auxWord = nltk.word_tokenize(patterns) ##Word tokenize reconoce puntos especiales y no solo las palabras.
        words.extend(auxWord)
        auxX.append(auxWord)
        auxY.append(content["tag"])

        if content["tag"] not in tags:
            tags.append(content["tag"])

words = [stemmer.stem(w.lower()) for w in words if w!='?' or w!='!']
words = sorted(list(set(words)))
tagas = sorted(tags)

training    = []
output      = []
emptyOutput = [0 for _ in range(len(tags))]

for x, file in enumerate(auxX):
    socket  = []
    auxWord = [stemmer.stem(w.lower()) for w in file]
    for w in words:
        if w in auxWord:
            socket.append(1)
        else:
            socket.append(0)
    rowsOutput = emptyOutput[:]
    rowsOutput[tags.index(auxY[x])] = 1
    training.append(socket)
    output.append(rowsOutput)

training = numpy.array(training)
output   = numpy.array(output)

tensorflow.Graph() ## Poner en blanco la red neuronal

red = tflearn.input_data(shape=[None, len(training[0])]) ## Va hacer el entrenamiento!
red = tflearn.fully_connected(red, 10) ## Con las siguientes neuronas va estar completamente conectado!
red = tflearn.fully_connected(red, 10) ## Se va a conectar a otras 10 redes neuronales
red = tflearn.fully_connected(red,len(output[0]), activation="softmax") ## Va a salir a los tags que en este caso son dos!
red = tflearn.regression(red) ## Poder obtener probabilidades!

model = tflearn.DNN(red)
model.fit(training, output, n_epoch = 1000, batch_size = 13, show_metric = True) ## n_epoch la cantidad de veces que va a ver el modelo, batch_size es cuantas entradas va a usar
model.save("model.tflearn")


