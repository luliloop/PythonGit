#!/usr/bin/env python
# coding: utf-8

# # Warm Up Python
# 
# Para un proyecto de ML es importante realizar una análisis previo de los datos con el fin de alistarlos para la fase de modelamiento.
# 
# ## Objetivo:
# El objetivo de este taller es aprender códigos básicos que te permitiran abordar en tus proyectos personales la fase de preprocesamiento y exploración de los datos.

# ## Exploración de Datos
# Para esta fase inicial conocer algunas funciones que nos provee pandas para facilitarnos la vida son fundamentales.  Dentro de esta fase se desea identificar el dataset que tipo de datos maneja, si tenemos nulos en nuestro conjunto de datos, entre otros.
# 
# Estas son algunas funciones que nos sirven para esta fase:
# 
# ```python
# df.head()
# df.tail()
# df.info()
# df.shape
# df.columns
# df.describe()
# df.value_counts()
# df.unique()
# df.nunique()
# ```
# 
# Trabajemos con el famoso conjunto de datos del [titanic](https://www.kaggle.com/c/titanic/data).
# 
# - PassengerId -- A numerical id assigned to each passenger.
# - Survived -- Whether the passenger survived (1), or didn't (0).
# - Pclass -- The class the passenger was in.
# - Name -- the name of the passenger.
# - Sex -- The gender of the passenger -- male or female.
# - Age -- The age of the passenger.  Fractional.
# - SibSp -- The number of siblings and spouses the passenger had on board.
# - Parch -- The number of parents and children the passenger had on board.
# - Ticket -- The ticket number of the passenger.
# - Fare -- How much the passenger paid for the ticket.
# - Cabin -- Which cabin the passenger was in.
# - Embarked -- Where the passenger boarded the Titanic.

# In[1]:


# Importando librerias
import pandas as pd
import seaborn as sb
# Cargamos el archivo
titanic = sb.load_dataset("titanic")
titanic = pd.read_csv('train.csv')


# In[6]:


# Visualiza las primeras 2 filas del titanic
titanic[:2]


# In[23]:


titanic.head(20)


# In[7]:


# Visualiza las ultimas 2 filas del titanic
titanic.tail(2)


# In[8]:


# Conocer cuantas filas y columnas tiene nuestro dataset
titanic.shape


# In[9]:


# Tipos de datos que tiene nuestro dataset y que cantidad de nulos tiene
titanic.info()


# In[15]:


# Otra manera de visualizar los nulos por columna
titanic.isnull().sum()
#display(titanic.isnull()) this shows data more esthetic


# In[16]:


# Otra manera de conocer los tipos de datos
titanic.dtypes


# In[17]:


# Resumen de las columnas numericas
titanic.describe()


# In[20]:


# Resumen de las columnas categoricas
import numpy as np
titanic.describe(include = [np.number])


# In[21]:


# Resumen de las columnas categoricas
import numpy as np
titanic.describe(include = ['O'])


# In[22]:


# Resumen de las columnas categoricas
import numpy as np
titanic.describe(include = [np.int64])


# In[14]:


# Valores unicos por columna sin tener en cuenta nulos
titanic.nunique()


# In[27]:


# Valores unicos por columna teniendo en cuenta nulos
titanic.nunique(dropna=False)


# In[32]:


# Conteo de valores por columna de no nulos
titanic.count()


# In[37]:


# Valor mas frecuente en la serie
titanic['Embarked'].mode()[0]


# In[38]:


# Valores unicos de la columna 'Embarked'
titanic['Embarked'].unique()


# In[39]:


# Cuantos registros tenemos por categoria de la columna 'Embarked' sin nulos
titanic['Embarked'].value_counts()


# In[40]:


# Cuantos registros tenemos por categoria de la columna 'Embarked' con nulos
titanic['Embarked'].value_counts(dropna=False)


# In[22]:


# Cuales son los nombres de las columnas del dataset
titanic.columns


# Ahora que tenemos una idea de la información que tenemos, vamos a empezar a transformarla para poder tener nuestros datos listos para la fase de visualización, de esta fase podemos concluir que PassengerId y Name que tienen 891 registros unicos no son columnas que nos sirvan para un analisis posterior, la columna Ticket tambien son 681 valores unicos esta tampoco nos sirve, y por ultimo la columna Cabin tiene el 77% de los datos nulos, lo que hace que tampoco nos sirva para más adelante.

# ### Practica
# 
# Con las columnas Genero ('Sex'), Edad ('Age') y el valor del tiquete ('Fare'), responde las siguientes preguntas:
# 
# - ¿Cuantos Hombres y Mujeres tenemos en nuestro dataset?
# - ¿Cual es la edad mas frecuente en nuestro conjunto de datos?
# - ¿Cual es la edad mas frecuente en las mujeres?
# - ¿Cual es la edad mas frecuente en los hombres?
# - ¿Cual es el precio del tiquete mas común?
# - ¿El precio del tiquete mas común es el mismo para hombres y para mujeres?

# In[23]:


#¿Cuantos Hombres y Mujeres tenemos en nuestro dataset?
titanic['Sex'].value_counts()


# **Deberia de tener el siguiente resultado:**
# 
# ```
# male      577
# female    314
# Name: Sex, dtype: int64
# ```

# Doble-clic __Aqui__ para la solución.
# 
# <!-- Your answer is below:
# titanic['Sex'].value_counts()
# -->

# In[42]:


#¿Cual es la edad mas frecuente en nuestro conjunto de datos?
titanic['Age'].mode()
titanic['Age'].value_counts()


# **Deberia de tener el siguiente resultado:**
# 
# ```
# 0    24.0
# dtype: float64
# ```

# Doble-clic __Aqui__ para la solución.
# 
# <!-- Your answer is below:
# titanic['Age'].mode()
# -->

# In[43]:


#¿Cual es la edad mas frecuente en las mujeres?
titanic[titanic['Sex']=='female']['Age'].mode()


# **Deberia de tener el siguiente resultado:**
# 
# ```
# 0    24.0
# dtype: float64
# ```

# *Doble*-clic __Aqui__ para la solución.
# 
# <!-- Your answer is below:
# titanic[titanic['Sex']=='female']['Age'].mode()
# -->

# In[44]:


#¿Cual es la edad mas frecuente en los hombres?
titanic[titanic['Sex']=='male']['Age'].mode()


# **Deberia de tener el siguiente resultado:**
# 
# ```
# 0    19.0
# 1    25.0
# 2    28.0
# dtype: float64
# ```

# Doble-clic __Aqui__ para la solución.
# 
# <!-- Your answer is below:
# titanic[titanic['Sex']=='male']['Age'].mode()
# -->

# In[45]:


#¿Cual es el precio del tiquete mas común?
titanic['Fare'].mode()


# **Deberia de tener el siguiente resultado:**
# 
# ```
# 0    8.05
# dtype: float64
# ```

# Doble-clic __Aqui__ para la solución.
# 
# <!-- Your answer is below:
# titanic['Fare'].mode()
# -->

# In[46]:


#¿El precio del tiquete más común es el mismo para hombres y para mujeres?

print('Precio común del tiquete para mujeres :' + str(titanic[titanic['Sex']=='female']['Fare'].mode()[0]))
print('Precio común del tiquete para hombres :' + str(titanic[titanic['Sex']=='male']['Fare'].mode()[0]))


# **Deberia de tener el siguiente resultado:**
# 
# ```
# Precio común del tiquete para mujeres :7.75
# Precio común del tiquete para hombres :8.05
# ```

# Doble-clic __Aqui__ para la solución.
# 
# <!-- Your answer is below:
# print('Precio común del tiquete para mujeres :' + str(titanic[titanic['Sex']=='female']['Fare'].mode()[0]))
# print('Precio común del tiquete para hombres :' + str(titanic[titanic['Sex']=='male']['Fare'].mode()[0]))
# -->

# ## Preprocesamiento:
# Para esta fase tener un buen conocimiento de Pandas será muy útil, adicional estructurar esta fase de preprocesamiento con la ayuda de funciones, nos permitirá tener codigo que podremos reutilizar en otros proyectos.
# 
# Dentro de las actividades que se podrían hacer en preprocesamiento se encuentras las siguientes:
# 
# **Borrar columnas**
# 
# **Enriching (or Transforming)** un conjunto de datos, agregando columnas recién calculadas en los índices.
# 
# **Filtering** seleccionando un subconjunto de las filas o columnas de un conjunto de datos de acuerdo con algún criterio.
# 
# **Indexing** agregando índices a un conjunto de datos.
# 
# **Aggregating**
# 
# **Sorting** ordenar las filas de un conjunto de datos según algún criterio
# 
# **Merging** fusionar los conjuntos de datos de alguna manera. Esto incluye: concatenación (horizontal o vertical) y también unión.
# 
# **Pivoting**: esto incluye la transposición y la realización de otras operaciones para que los datos que originalmente tenían un diseño vertical se distribuyan horizontalmente (aumentando el número de columnas) o viceversa (aumentando el número de filas). 
# 
# Entre otras.

# In[47]:


# Eliminar columnas PassengerId, Name, Cabin, la columna ticket no la vamos a eliminar todavia.
titanic.drop(['PassengerId','Name','Cabin'],axis=1, inplace=True)
titanic.shape


# In[48]:


# Validar nulos nuevamente
titanic.isnull().sum()


# In[49]:


# En los primeros 10 registros tenemos un campo de edad nulo, vamos a tenerlo como referencia para lo que vamos a hacer
titanic.head(10)


# ### Nulos
# Dependiendo del problema el tratamiento de los nulos se puede manejar de diferentes maneras:
# 
# * Conservalos
# * Eliminarlos
# * Dejar un valor fijo que los diferencie
# * Reemplazarlos con un estadistico (media, mediana, moda) Cual usar si tenemos outliers?
# * Forward o backward fill

# In[50]:


# Para eliminarlos usamos
titanic.dropna(inplace=True)
titanic.shape


# In[51]:


# Validar nulos nuevamente
titanic.isnull().sum()


# In[52]:


# En los primeros 10 registros tenemos un campo de edad nulo, vamos a tenerlo como referencia para lo que vamos a hacer
titanic.head(10)


# In[53]:


# Dejar un valor fijo que los diferencie
titanic = pd.read_csv("train.csv")
titanic.drop(['PassengerId','Name','Cabin'],axis=1, inplace=True)
titanic.fillna(-1).head(10)


# In[54]:


titanic['Age'].fillna(titanic['Age'].mean()).head(10)


# In[55]:


titanic['Age'].fillna(titanic['Age'].median()).head(10)


# In[56]:


titanic['Age'].fillna(titanic['Age'].mode()[0]).head(10)


# In[57]:


titanic['Age'].fillna(method='backfill').head(10)


# In[40]:


titanic['Age'].fillna(method='ffill').head(10)


# Para nuestro ejercicio tomaremos la decisión de borrar los nulos.

# In[41]:


# Para eliminarlos usamos
titanic.dropna(inplace=True)
titanic.shape


# ### Columnas
# Para ser practicos, normalmente modificamos los nombres de las columnas con el fin de quitar espacios en blanco, caracteres especiales, dejar los nombres en minuscula así que a continuacion realizaremos ese cambio.

# In[42]:


# Cambiar nombre de columnas a minusculas
titanic.rename(columns = lambda col: col.lower(),inplace=True)


# In[43]:


# Validar dataset
titanic.info()


# ### Indices

# In[44]:


# Luego de eliminar los nulos, se debe de resetear los indices, como puedes ver aparecen de 0 a 890 todavia 
# a persar de que tenemos 712 filas.
titanic.reset_index(drop=True,inplace=True)
titanic.info()


# ### Agrupación

# In[45]:


# Validar los tickets duplicados
df_tickets = titanic.groupby('ticket').size().reset_index().rename(columns={0:'count'})
df_tickets.head()


# ### Ordenando

# In[46]:


# Ordenar los cantidad de registros por tickets
df_tickets.sort_values(by=["count"],ascending=False,inplace=True)
df_tickets.head()


# In[47]:


# Validemos un tiquete
titanic[titanic['ticket']=='347082']


# ### Uniendo datasets
# Uno de las tareas que se realizan en preprocesamiento es enriquecer la información, crearemos una nueva columna llamada 'family' en esta realizaremos una marcación de 1 si el viaje fue familiar o viajaron solos.

# #### Merge and join
# 
# Ambas funciones permiten que los datos de diferentes dataframes se combinen en uno solo de acuerdo con una regla de "cruce" o "búsqueda".
# 
# Aunque tanto `merge` como` join` hacen cosas similares, la forma en que lo hacen es diferente.
# 
# La función `merge` es la función predeterminada de pandas para unir datos. Básicamente es contraparte de *pandas de la unión de SQL*, y requiere la especificación de qué columnas de ambos dataframes se compararán. A Merge no le importa en absoluto los índices definidos en ellos.
# 
# Por otro lado, la función `join` de Panda es más conveniente (incluso utiliza merge internamente), unir es básicamente hacer una fusión aprovechando los índices de ambos marcos de datos.
# 
# La siguiente figura resume los diferentes 4 tipos de combinaciones: _inner, outer, left and right_.
# 
# <img src="https://github.com/darkanita/MLCourse/blob/master/Introduccion/merge.png?raw=1"/>
# 
# La función merge también está disponible como método en la clase `DataFrame`.
# La sintaxis básica es:
# 
# ```
# new_joined_df = df.merge (another_df, left_on = "col_in_df", right_on = "col_in_another_df",
#                           how="inner"|"left"|"right"|"outer")
# ```
# 
# El primer argumento (`another_df`), así como` left_on` y `right_on` son argumentos obligatorios.
# `left_on` especifica un nombre de columna en el dataframe `df` cuyos valores deben coincidir con
# los de la columna `another_df` 'especificados en `right_on`.
# 
# El argumento `how` es opcional y por defecto es `inner`.

# In[48]:


# Primero adicionaremos la columna 'count' de df_tickets a nuestro dataset titanic
titanic = titanic.merge(df_tickets,left_on='ticket',right_on='ticket',how='left')
titanic.shape


# In[49]:


titanic.head()


# ### Enriquecer dataset

# In[50]:


titanic['family'] = [1 if count>1 else 0 for count in titanic['count']]


# In[51]:


titanic.head()


# ### Limpliando Cadenas

# In[52]:


# Recuerdan las funciones? vamos a crear una funcion que nos servira para limpiar texto, en este caso el campo tickets
def limpiar_caracteres(cadena):
    import re
    patron = '[^A-Za-z0-9]+'
    return re.sub(patron, '', cadena)


# In[53]:


titanic['ticket'] = titanic['ticket'].apply(limpiar_caracteres)
titanic.head()


# In[54]:


# Para continuar eliminaremos la columna ticket
titanic.drop('ticket',axis=1,inplace=True)
titanic.head()


# ### Codificando variables categoricas
# 
# En ocasiones se requiere volver columnares nuestros datos categoricos, para esto usamos tambien un metodo de pandas llamado get_dummies.

# In[55]:


titanic['sexDummies'] = titanic['sex']
titanic = pd.get_dummies(titanic, columns = ['sexDummies'], prefix = ['D'])
titanic.head()


# Que hemos aprendido:
# 
# - Tratar Nulos
# - Transformar columnas
# - Enriquecer información en nuestro dataset adicionando columnas calculadas
# - Hacer agregaciones
# - Unir datasets.
# - Limpiar texto con la ayuda de Regex
# - Codificando variables categoricas

# ### Practica
# 
# - Agrupar el conjunto de datos por genero ('sex') y calcular el promedio de la edad ('age').
# - Agrupar el conjunto de datos por la columna familia ('family') y contar la cantidad de registros por grupo.
# - De las familias cuantos se sobrevivieron y cuantos no.

# In[56]:


# Agrupar el conjunto de datos por genero ('sex') y calcular el promedio de la edad ('age')
titanic.groupby('sex').agg({'age':'mean'}).reset_index()


#  **Deberia de tener el siguiente resultado:**
# 
# | |sex| age | 
# |-----|-----|-----|
# |0|	female|	27.915709|
# |1|male|30.726645|

# Doble-clic __Aqui__ para la solución.
# 
# <!-- Your answer is below:
# titanic.groupby('sex').agg({'age':'mean'}).reset_index()
# -->

# In[57]:


# Agrupar el conjunto de datos por la columna familia ('family') y contar la cantidad de registros por grupo.
titanic.groupby(['family']).size().reset_index().rename(columns={0:'count'})


#  **Deberia de tener el siguiente resultado:**
# 
# | |family| count | 
# |-----|-----|-----|
# |0|	0|	547|
# |1|1|344|

# Doble-clic __Aqui__ para la solución.
# 
# <!-- Your answer is below:
# titanic.groupby(['family']).size().reset_index().rename(columns={0:'count'})
# -->

# In[58]:


# De las familias cuantos se sobrevivieron y cuantos no.
titanic.groupby(['family','survived']).size().reset_index().rename(columns={0:'count'})


#  **Deberia de tener el siguiente resultado:**
# 
# | |family| survived | count |
# |-----|-----|-----|-----|
# |0|	0|	0| 384|
# |1|	0|	1| 163|
# |2|	1|	0| 165|
# |3|	1|	1| 179|

# Doble-clic __Aqui__ para la solución.
# 
# <!-- Your answer is below:
# titanic.groupby(['family','survived']).size().reset_index().rename(columns={0:'count'})
# -->

# ## Visualización:
# Para apoyar el análisis descriptivo de los datos usamos la exploración de estos a través de gráficos que nos permiten conocer más la información que vamos a utilizar para nuestros modelos, para esta fase es muy util tener conocimiento de las librerias de Python que nos permiter generar estas visualizaciones como Matplotlib y Seaborn, Pandas tambien tiene unos métodos de visualización. 

# In[59]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

titanic.head()


# ### Lineas

# In[60]:


# Empecemos con un grafico sencillo, al no colocar X se asume que x es cada uno de los registros.
plt.figure(figsize=(6,3))
plt.plot(titanic['age'])
plt.title('Titanic Graficando la Edad')
plt.show()


# ### Histogramas

# In[61]:


plt.figure(figsize=(6,4))
sns.distplot(titanic['age'])
plt.title('Titanic Histograma de la Edad')
plt.show()


# In[62]:


# Si solo queremos la funcion de densidad
plt.figure(figsize=(6,3))
sns.kdeplot(titanic['age'],shade=True)
plt.title('Titanic Densidad de la Edad')
plt.xlabel('age')


# In[63]:


g = sns.FacetGrid(titanic, col="survived", height=3)
g.map(sns.kdeplot, 'age', shade=True)
g.despine(left=True,bottom=True)


# In[64]:


g = sns.FacetGrid(titanic, col='survived', row='pclass', hue='sex', height=3)
g.map(sns.kdeplot, 'age', shade=True).add_legend()
g.despine(left=True, bottom=True)
plt.show()


# ### Boxplot

# In[65]:


plt.figure(figsize=(6,3))
sns.boxplot(titanic['age'],orient='v')
plt.title('Titanic Boxplot de la Edad')
plt.show()


# In[66]:


plt.figure(figsize=(6,4))
sns.boxplot(x='survived',y='age',data=titanic)
plt.title('Titanic Boxplot de los sobrevivientes y la edad')
plt.show()


# In[67]:


plt.figure(figsize=(10,4))
sns.boxplot(x='sex',y='age',hue='survived',data=titanic)
plt.title('Titanic Boxplot de los sobrevivientes, el genero y la edad')
plt.show()


# ### Barras

# In[68]:


plt.figure(figsize=(6, 4))
sns.countplot('survived',data=titanic)
plt.show()


# In[69]:


plt.figure(figsize=(6, 4))
sns.barplot(y='survived',x='pclass',data=titanic)
plt.show()


# In[70]:


plt.figure(figsize=(6, 4))
sns.barplot(y='survived',x='sex',hue='pclass',data=titanic)
plt.show()


# In[71]:


plt.figure(figsize=(6, 4))
sns.barplot(y='pclass',x='survived',hue='family',data=titanic)
plt.show()


# # Heatmap

# In[72]:


plt.figure()
sns.heatmap(titanic.drop(['survived','family','D_female','D_male','count'],axis=1).corr(),annot=True, linewidth=0.5,fmt='.1f')
plt.show()


# ### Scatter Plot

# In[73]:


plt.figure()
sns.scatterplot(x='age', y='fare', hue='sex', style='survived', data=titanic)
plt.show()


# In[74]:


plt.figure()
sns.scatterplot(x='age', y='fare', hue='sex', style='survived',size='count', data=titanic)
plt.show()


# In[75]:


plt.figure()
sns.scatterplot(x='age', y='fare',data=titanic)
plt.show()


# ### Practicar
# 

# El Departamento de Estadísticas de Educación publica anualmente un conjunto de datos que contiene el porcentaje de títulos de licenciatura otorgados a mujeres de 1970 a 2012. El conjunto de datos se divide en 17 categorías de títulos, con cada columna como una categoría separada.
# 
# Randal Olson, científico de datos de la Universidad de Pensilvania, limpió el conjunto de datos y lo puso a disposición en su sitio web personal. Puede descargar el conjunto de datos Randal compilado [aquí](http://www.randalolson.com/wp-content/uploads/percent-bachelors-degrees-women-usa.csv). Aquí hay una vista previa de las primeras filas:

# In[63]:


import pandas as pd
import matplotlib.pyplot as plt


# In[68]:


women_degrees = pd.read_csv('percent-bachelors-degrees-women-usa.csv')
women_degrees.head()


# Randal compiló este conjunto de datos para explorar la brecha de género en los campos STEM, que significa ciencia, tecnología, ingeniería y matemáticas. Esta brecha se informa a menudo en las noticias y no todos están de acuerdo en que existe una brecha.
# 
# El objetivo de esta practica es explorar esta información y validar si se evidencia alguna brecha de género utilizando una visualización de datos efectiva. Primero generemos un diagrama matplotlib estándar.

# In[69]:


plt.plot(women_degrees["Year"],women_degrees["Biology"])
plt.show()


# In[70]:


women_degrees.info()


# In[71]:


plt.plot(women_degrees["Year"],women_degrees["Biology"],color='blue',label="Women")
plt.plot(women_degrees["Year"],100-women_degrees["Biology"],color='red',label="Men")
plt.title("Percentage of Biology Degrees Awarded By Gender")
plt.legend(loc="upper right")
plt.tick_params(bottom="off",top="off",left="off",right="off")
plt.show()


# In[ ]:




