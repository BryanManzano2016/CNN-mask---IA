from tkinter import *
from PIL import ImageTk, Image
from tkinter import filedialog

import numpy as np
from tensorflow import keras
import tensorflow as tf
from tensorflow.python.keras.models import load_model

from tkinter import messagebox

longitud, altura = 200, 200
modelo = './modelo_project_modelo_20200903_211740/modelo.h5'
pesos_modelo = './modelo_project_modelo_20200903_211740/pesos.h5'
cnn = load_model(modelo)
cnn.load_weights(pesos_modelo)


def open_img():
    x = openfilename()

    img = Image.open(x)
    img = img.resize((300, 300), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)
    panel = Label(root, image=img)
    panel.image = img
    panel.grid(row=2)

    img = keras.preprocessing.image.load_img(
        x, target_size=(altura, longitud)
    )
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions = cnn.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    class_names = ['CON MASCARILLA', 'SIN MASCARILLA']
    messagebox.showinfo("CNN mask", "Imagen: {} \nTipo: {} \nPorcentaje de acierto: {:.2f}."
                        .format(x, class_names[np.argmax(score)], 100 * np.max(score)))


def openfilename():
    filename = filedialog.askopenfilename(title='"Seleccione la imagen')
    return filename


root = Tk()

root.title("CNN Covid mask")
root.geometry("350x350")
root.resizable(width=True, height=True)

btn = Button(root, text='Cargar una imagen', command=open_img).grid(row=1)

root.mainloop()
