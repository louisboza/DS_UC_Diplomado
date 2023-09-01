####################################
##                                ##
## Clase 17 - Árboles de decisión ##----
##                                ##
####################################

# En esta clase revisaremos el primer modelo de Machine Learning que veremos, 
# llamado Árbol de decisión. Los Árboles son modelos bastante versátiles y 
# potentes en el mundo del Machine Learning, los cuales presentan las siguientes 
# características:
#   
# - Se pueden utilizar para problemas de clasificación, regresión u otros
# - No requiere escalar las variables de ingreso
# - Alta interpretabilidad
# - Forman la base de modelos más sofisticados (Random Forest, entre otros)
# 
# Para mostrar lo que podemos hacer con un árbol de decisión, utilizaremos un 
# set de datos sobre los precios de inmuebles en la ciudad de Ames, Iowa. 
# La base se compone de 2930 registros y contiene un gran número de atributos. 
# Nuestro objetivo es generar un modelo que prediga de forma adecuada los precios 
# de inmuebles, medidos con la variable Sale_Price (variable numérica)

#Cargamos los packages
require(dplyr)
require(data.table)
require(tidyverse)
require(caret)
require(rattle)
require(rpart) # para realizar los árboles
require(Metrics)
require(tidyverse)
require(ipred)


#Cargamos los datos
ames <- read.csv('ames_housing.csv')
head(ames)

#Lo primero es ver que tipo de datos son, antes de poder manipular los datos.
glimpse(ames)

chr_cols <- ames %>%
  select_if(is.character)
chr_cols <- colnames(chr_cols)

ames <- ames %>%
  mutate_each_(funs(factor(.)),chr_cols)
glimpse(ames)


# Además de lo anterior, la primera columna es un indicador, y además la latitud 
# y longitud tipicamente no entregan información relevante tal y como están, 
# por lo que las eliminaremos

ames <- ames %>%
  select(-X,-Longitude,-Latitude)


# Arboles de decision -----------------------------------------------------

# Realizamos un modelo solo con algunas variables SOLO
# por conveniencia y para poder visualizarlo bien

model <- rpart(Sale_Price ~ Sale_Condition +
                 Pool_QC + Enclosed_Porch + Fence, data = ames)

# Paquete para realizar la visualizacion del arbol

library(rpart.plot)
rpart.plot(model)

glimpse(ames)
ames$sale_price_bin <- 0

ames$sale_price_bin[ames$Sale_Price > 180000] = 1
ames$sale_price_bin = as.factor(ames$sale_price_bin)

model_2 <- rpart(sale_price_bin ~ Sale_Condition +
                 Pool_QC + Enclosed_Porch + Fence + 
                 Garage_Area, 
                 data = ames)

rpart.plot(model_2, cex = 0.5)
# cex = 0.5 -> ajusta el tamaño del arbol (solo visualmente)


# Modelos Bagging ---------------------------------------------------------
#    B     +  agging
# aggregating + boostrapping
# Boostrap + Agregation 
# Boostrap -> toma muestra de los datos con reemplazo, crea un nuevo conjunto
# de los datos ("re-muestreando"):
#
# conjunto_1 = c(1, 2, 3, 4, 5)
# muestra_boostrap_tamaño4 = c(2, 3, 3, 4) 
#
#
# Ahora que tenemos un entendimiento de los árboles, abordaremos el concepto 
# de Bagging
#   
# Este método consiste en generar replicas del dataset de entrenamiento mediante 
# la técnica de bootstrap, de esta forma podremos entrenar nuestro modelo con 
# distintos dataset tomando como predicciones en ellos ya sea un método de 
# votación (en el caso de clasificación) o un promedio (en caso de regresión) 
# entre todos.
# 
# La formulación de este método permite claramente generarlo para cualquier 
# modelo, pero el modelo donde más popular se ha hecho este método es 
# utilizandolo sobre árboles de decisión, los que se conocen como Random Forest.

# Para realizar bagging

require(randomForest)

train_set <- ames[1:2344, ]
test_set <- ames[2345:2930, ]

modelo_bagging <- randomForest(sale_price_bin ~ Sale_Condition +
                               Pool_QC + Enclosed_Porch + 
                               Fence + 
                               Garage_Area, 
                               data = train_set,
                               ntree = 100, # Crea 100 arboles (*)
                               mtry = 5) 
# (*) Por lo que toma 100 muestras con reemplazo para ajustar cada arbol.

# mtry: cantidad de variables que vamos a utilizar en el modelo, por lo que
# si tenemos 5 variables en total para predecir y escogemos mtry = 5,
# entonces estamos realizando bagging pues estamos utilizando las mismas
# 5 variables para realizar los arboles.
                               

# Vemos la importancia de cada variable:

randomForest::importance(modelo_bagging)

# Realizamos las predicciones

prediccion_mod_bag = predict(modelo_bagging, newdata = test_set)


# Ahora realizaremos un modelo basado en Random forest:

# Agrega un paso mas a parte de aggregating + boostrapping, toma aleatoriamente 
# covariables (remuestrea) para cada arbol por lo que genera arboles mas 
# incorrelacionados entre si
# Ejemplo: 
#
# Data set:
#  |X1|X2|X3|X4| Y |
# 1|  |  |  |  |   |
# 2|  |  |  |  |   |
# 3|  |  |  |  |   |
# 4|  |  |  |  |   |
# 5|  |  |  |  |   |
# 6|  |  |  |  |   |
#
# Para un arbol, tomamos la siguiente tabla:
# Tabla 1:

#  |X2|X4| Y |
# 2|  |  |   |
# 2|  |  |   |
# 3|  |  |   |
# 6|  |  |   |

# Tomamos ALEATORIAMENTE las variables X2 y X4 y ADEMAS tomamos una muestra de 
# las observaciones, en este caso se selecciono la fila 2, 3 y 6, en donde, 
# como la muestra es con reemplazo, la fila 2 se repite dos veces.
# Entonces, si tenemos 100 arboles, entonces debemos crear 100 tablas 


modelo_random <- randomForest(sale_price_bin ~ Sale_Condition +
                              Pool_QC + Enclosed_Porch + 
                              Fence + 
                              Garage_Area, 
                              data = train_set,
                              ntree = 100,
                              # mtry = sqrt(p)
                              ) 
# Modelo de arbol

modelo_arbol <- rpart(sale_price_bin ~ Sale_Condition +
                      Pool_QC + Enclosed_Porch + Fence + 
                      Garage_Area, 
                      data = train_set)

# Ahora veremos que modelo tiene mejor rendimiento utilizando
# el set de testeo

# Predicciones
?predict.rpart
prediccion_arbol = predict(modelo_arbol, newdata = test_set,
                           type = 'class') # utiliza como punto de corte 0.5
prediccion_bagging = predict(modelo_bagging, newdata = test_set)
prediccion_random = predict(modelo_random, newdata = test_set)

valor_real = test_set$sale_price_bin

# matriz de confusion

table(prediccion_arbol, valor_real)
# Sensibilidad = 147/(147+63) = 0.7
# Especificidad = 298/(298 +78) = 0.79

table(prediccion_bagging, valor_real)
# Sensibilidad = 0.67
# Especificidad = 307/(307+69) = 0.816

table(prediccion_random, valor_real)
# Sensibilidad = 0.695
# Especificidad = 301/(301 + 75) = 0.8


