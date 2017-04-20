# Prueba 7-04-2017
# Código de la entrada a kaggle con mejor puntuación obtenida

source("librerias.R")
# Exportación del dataset
imb.train <- read.csv("./datos/pv1math-tra.csv",sep = ",", header = TRUE, stringsAsFactors = FALSE)
imb.test <- read.csv("./datos/pv1math-tst.csv", sep = ",", header = TRUE, stringsAsFactors = FALSE)

# Eliminamos el id del test, añadimos la variable clase a test y los unimos para realizar el preprocesamiento de un paso
imb.test[["ID"]] <- NULL
imb.train$PV1MATH <- as.factor(imb.train$PV1MATH)
imb.test$PV1MATH <- NA

###################################
#  Previsualización de los datos  #
###################################
str(imb.train)

##############
#  Outliers  #
##############

# deteccion de anomalias para las variable 7 y 9 a 22.
# La funcion devuelve el valor (o valores)
# considerados anomalos para las variable de interes. Este
# metodo solo considera las desviaciones con respecto a los
# valores de cada variable (no relaciones con otras variables)
anomalos <- outlier(imb.train[,c(7,9:22)])
print(anomalos)
# Variable ESC
plot(imb.train$ESCS, col=imb.train$PV1MATH)
which(imb.train$ESCS <= -4.28)
imb.train <- imb.train[-2010,]
# Variable BELONG
plot(imb.train$BELONG, col=imb.train$PV1MATH)
which(imb.train$BELONG <= -3)
imb.train <- imb.train[-c(127,196),]
# Variable STUDREL
plot(imb.train$STUDREL, col=imb.train$PV1MATH)
which(imb.train$STUDREL <= -3.11)
imb.train <- imb.train[-c(312, 920, 1066, 1163, 1335, 2434, 2438, 2442, 2455)]
# Variable ATTLNACT
plot(imb.train$ATTLNACT, col=imb.train$PV1MATH)
which(imb.train$ATTLNACT <= -3.3758)
imb.train <- imb.train[-c(1137,1160,1200,1216,1287,1335,2324,2434),]
# Variable ATSCHL
plot(imb.train$ATSCHL, col=imb.train$PV1MATH)
which(imb.train$ATSCHL <= -2.99)
imb.train <- imb.train[-c(1178, 1532, 2430, 3013),]
# Variable ANXMAT
plot(imb.train$ANXMAT, col=imb.train$PV1MATH)
sum(imb.train$ANXMAT <= -2.37) # No quito ninguno demasiados
# Variable FAILMAT
plot(imb.train$FAILMAT, col=imb.train$PV1MATH)
which(imb.train$FAILMAT >= 3.9067) # Demasiados, no se quita ninguno
# Variable INSTMOT
plot(imb.train$INSTMOT, col=imb.train$PV1MATH)
which(imb.train$INSTMOT <= -2.3) # Demasiados no se quita ninguno
# Variable INTMAT
plot(imb.train$INTMAT, col=imb.train$PV1MATH)
which(imb.train$INTMAT >= 2.29) # Demasiados
# Variable SMATBEH
plot(imb.train$SMATBEH, col=imb.train$PV1MATH)
which(imb.train$SMATBEH >= 4.4249)
imb.train <- imb.train[-c(202, 800, 1215, 1480, 2250, 2280, 2586, 2833, 2928, 2978),]
# Variable MATHEFF
plot(imb.train$MATHEFF, col=imb.train$PV1MATH)
which(imb.train$MATHEFF <= -3.7500)
imb.train <- imb.train[-c(340,434,980,984,1150,1162,1268,1300,1321,1462,1806,2150,2953),]
# Variable MATINTFC
plot(imb.train$MATINTFC, col=imb.train$PV1MATH)
which(imb.train$MATINTFC <= -1.5329)
# Variable MATWKETH
plot(imb.train$MATWKETH, col=imb.train$PV1MATH)
which(imb.train$MATWKETH <= -3.4503)
imb.train <- imb.train[-c(385, 467, 622, 737, 871, 1179, 1278, 1433, 2512, 2631, 2635, 2659, 2685, 2698, 2797, 2943),]
# Variable SCMAT
plot(imb.train$SCMAT, col=imb.train$PV1MATH)
which(imb.train$SCMAT <= -2.18)
# Variable SUBNORM
plot(imb.train$SUBNORM, col=imb.train$PV1MATH)
which(imb.train$SUBNORM <= -4.2456)
imb.train <- imb.train[-c(847, 2592, 2701, 2749, 3121),]

######################
#  Normalizar datos  #
######################

# Se juntan train y test para normalizar los datos
nTrain <- nrow(imb.train)
nTest <- nTrain + 1
full.imb <- rbind(imb.train, imb.test)
valoresPreprocesados <- caret::preProcess(full.imb[,c(7,9:22)],method=c("range"))
valoresTransformados <- predict(valoresPreprocesados, full.imb[,c(7,9:22)])
full.imb[,c(7,9:22)] <- valoresTransformados
# Se vuelven a separar
imb.train <- full.imb[1:nTrain,]
imb.test <- full.imb[nTest:nrow(full.imb),]
imb.test$PV1MATH <- NULL

############################
#  Selección de variables  #
############################

# mrMR
ind <- sapply(imb.train, is.integer)
imb.train[ind] <- lapply(imb.train[ind], as.numeric)
imb.train$PV1MATH <- as.numeric(imb.train$PV1MATH)
dd <- mRMR.data(data = imb.train)
feats <- mRMR.classic(data = dd, target_indices = c(ncol(imb.train)), feature_count = 23)
variableImportance <-data.frame('importance'=feats@mi_matrix[nrow(feats@mi_matrix),])
variableImportance$feature <- rownames(variableImportance)
row.names(variableImportance) <- NULL
variableImportance <- na.omit(variableImportance)
variableImportance <- variableImportance[order(variableImportance$importance, decreasing=TRUE),]
print(variableImportance)

# Volvemos a pasar la variable clase a factor
imb.train$PV1MATH <- as.factor(imb.train$PV1MATH)

# Consistency
# se usa el metodo consistency para seleccionar el subconjunto
# de atributos. Este metodo usa a su vez la funcion best.first.search
# para determinar el subconjunto mas prometedor
subset <- consistency(PV1MATH~., imb.train)
# se muestra el resultado
f <- as.simple.formula(subset,"PV1MATH")
print(f)

# Random Forest
# se calculan los pesos
weights <- FSelector::random.forest.importance(PV1MATH~.,imb.train, importance.type=1)
# se muestran los resultados
print(weights)
subset <- cutoff.k(weights,5)
f <- as.simple.formula(subset,"PV1MATH")
print(f)

# Prueba quitando las variables: 14, 2, 13
imb.train <- imb.train[,-c(14,2,13)]
imb.test <- imb.test[,-c(14,2,13)]


##############################
#  Particiones de los datos  #
##############################

trainData <- imb.train

###########
#  SMOTE  #
###########
#library(unbalanced)

trainData$PV1MATH <- as.factor(trainData$PV1MATH)
n <- ncol(trainData)
output <- trainData[,"PV1MATH"]
input <- trainData[,-n]
data <- ubSMOTE(X=input, Y= output)
newData <- cbind(data$X, data$Y)
# Volver a poner el nombre de la última columana
names(newData)[n] <- "PV1MATH"
table(newData$PV1MATH)
prop.table(table(newData$PV1MATH))

###########
# Modelos #
###########

rf.smote <- randomForest::randomForest(PV1MATH~., newData, ntree=250)
# Creamos la predicción para la entrega en kaggle
prediction <- predict(rf.smote, imb.test, decision.values = TRUE , type="prob")
imb.test$id <- seq.int(nrow(imb.test))
# En kaggle me da el mismo resultado (tanto la columna 1 como la 2)
submit <- data.frame(ID = imb.test$id, Prediction = prediction[,1])
View(submit)
# Acordarse poner el nombre bien de la columna "predict"
write.csv(submit, file = "resultados/result_7A.smote.rf.csv", row.names = FALSE)
