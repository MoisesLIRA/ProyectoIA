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

# Lee el archivo CSV y carga los datos en un DataFrame de pandas
DataFrame = pd.read_csv('datasetIA.csv', sep=';')
print(DataFrame.head())


# Obtener las columnas que tienen información
columnas_con_info = [columna for columna in DataFrame.columns if DataFrame[columna].notna().any()]

# Imprimir las columnas con información
print("Columnas con información:")
for columna in columnas_con_info:
    print(f"- {columna}")
    print(DataFrame[columna].sample(3)) 
    print()


# Obtener estadísticas descriptivas
print("Estadísticas descriptivas:")
print(DataFrame.describe())
print("\n")