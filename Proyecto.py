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

# Verificar valores faltantes
print("\nValores faltantes por columna:")
print(DataFrame.isnull().sum())

# Obtener valores únicos de la columna Sentimientos
# Nos interesa esta columna para clasificar los datos
columna = 'Sentimiento'
print(f"\nValores únicos en la columna '{columna}':")
print(DataFrame[columna].unique())


# Eliminar columnas no necesarias
dataset = DataFrame['Unnamed: 10', 'Unnamed: 11', 'Unnamed: 12'], axis=1, errors='ignore'

# Manejar valores faltantes en la columna 'Etiquetas de Trastorno Psicológico'
dataset['Etiquetas de Trastorno Psicológico'] = dataset['Etiquetas de Trastorno Psicológico'].fillna('Sin etiqueta')

# Manejar valores faltantes en la columna 'Anotaciones Clinicas'
dataset['Anotaciones Clinicas'] = dataset['Anotaciones Clinicas'].fillna('Sin anotación')

# Después de las transformaciones
print("\nDataFrame después de las transformaciones:")
print(dataset)