# Importarmos las biblitecas a utilizar
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Descargar los recursos de NLTK
nltk.download('stopwords')
nltk.download('punkt')


# Configurar para mostrar todas las columnas
pd.set_option('display.max_columns', None)

# Lee el archivo CSV y carga los datos en un DataFrame de pandas
DataFrame = pd.read_csv('datasetIA.csv', sep=';')

# Visualizar las primeras filas del dataset para comprender su estructura
print(DataFrame.head())

# Obtenemos la información sobre las columnas y tipos de datos
print(DataFrame.info())

import re
def preprocesamientoTexto(texto):
    # Eliminar caracteres especiales
    texto = re.sub(r'[^a-zA-Z0-9\s]', '', texto)

    # Convertir a minúsculas y tokenizar
    tokens = nltk.word_tokenize(texto.lower())

    # Eliminar stopwords y realizar stemming
    stemmer = PorterStemmer()
    stopwords_esp = set(stopwords.words('spanish'))
    palabras_procesadas = [stemmer.stem(palabra) for palabra in tokens if palabra not in stopwords_esp]

    # Unir las palabras procesadas en un texto
    texto_procesado = ' '.join(palabras_procesadas)

    return texto_procesado

# Aplicar preprocesamiento a la columna 'Texto'
DataFrame['Texto'] = DataFrame['Texto'].apply(preprocesamientoTexto)

# Visualizar algunos ejemplos de datos preprocesados
print("Ejemplos de datos preprocesados:")
print(DataFrame['Texto'].head(10))
