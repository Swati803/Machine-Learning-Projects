#################################################################################
#   Handwritten digits recognition based on the pixel values given as features	#
#   Support Vector Machines - Assignment                  						#
#   Swati Jaiswal			                              						#
#   PG Diploma in Data Analytics March 2017 			  						#
#################################################################################

####################################################################################
# 1. Business Understanding
# 2. Common utility functions used across this code
# 3. Data Understanding
# 4. Data Preparation
# 5. Model Building 
#   5.1 Linear SVM Model at C=1
#   5.2 Linear SVM Model at C=10
#   5.3 Non Linear model - SVM
# 6  Cross validation 
#   6.1. Tunning linear SVM model 
#   6.2. Tunning linear SVM model 
# 7  Conclusion/Summary 
####################################################################################
# 1. Business Understanding: 
#----------------------------

##Loading Neccessary libraries
#----------------------------

library(kernlab)
library(readr)
library(caret)
library(caTools)
library(e1071)

#Loading Data
#----------------------------
mnist_train <- read.csv("mnist_train.csv", header = F, stringsAsFactors = F)
mnist_test <- read.csv("mnist_test.csv", header = F, stringsAsFactors = F)

#####################################################################################
# 2. Common utility functions used across this code

# 2.1 the function prints the statistics of confusion matrix
#-----------------------------------------------------------
printConfusionMatrixStats <- function(confMat){
  total_accuracy_model <- 0
  sum_sensitivity_model <- 0
  sum_specificity_model <- 0
  
  for(i in 1:10){
    #str(cf_mat)
    sum_sensitivity_model <-  sum_sensitivity_model + confMat$byClass[i,1] 
    sum_specificity_model <-  sum_specificity_model + confMat$byClass[i,2]
  }
  avg_sensitivity_model <- sum_sensitivity_model/10
  avg_specificity_model <- sum_specificity_model/10
  total_accuracy_model <- confMat$overall[1]
  cat(sprintf("Total Accuracy of given svm model %f\n", total_accuracy_model))
  cat(sprintf("Average Sensitivity of svm model is %f and Specificity is %f\n", 
              avg_sensitivity_model , avg_specificity_model))
  
}

# 2.2 the function checks if variable exists
#--------------------------------------------
is.defined <- function(sym) {
  sym <- deparse(substitute(sym))
  env <- parent.frame()
  exists(sym, env)
}

#####################################################################################
# 3. Data Understanding
#----------------------------

cat(sprintf("Training set has %d rows and %d columns\n", nrow(mnist_train), ncol(mnist_train)))
cat(sprintf("Test set has %d rows and %d columns\n", nrow(mnist_test), ncol(mnist_test)))

#Understanding Dimensions
#----------------------------
dim(mnist_train) #60000 observations and 784 variables

#Structure of the dataset
#----------------------------
str(mnist_train) #10000 observations and 784 variables

#printing first few rows
#----------------------------
head(mnist_train)

#Exploring the data
#----------------------------
summary(mnist_train)

#NA values in the train dataset
#----------------------------
sum(is.na(mnist_train)) # Zero null values

#NA values in test dataset
#----------------------------
sum(is.na(mnist_test))  # Zero null values

#####################################################################################
# 4. Data Preparation
#----------------------------
# Changing output variable "lable" to factor type 
label <- as.factor(mnist_train[[1]]) 
test_label <- as.factor(mnist_test[[1]]) 
dftrain.pixel <- mnist_train[,2:ncol(mnist_train)]
dftest.pixel <- mnist_test[,2:ncol(mnist_test)]

# There are so many attributes/variables 784 in number, so here it makes sense to apply the dimensionality reduction or attribute
# reduction using factor rotations in factor analysis.
# Reduce the data set using PCA - principal component analysis technique of factor rotations in factor analysis

# first divide the pixel data set by 255, the reason for doing this is that
# the possible range of pixel is  0 to 255, so dividing the data set by 255
# brings the value of each pixel value to the range of [0-1] which is nrmalized scale.
Xtrain_reduced <- dftrain.pixel/255
Xtrain_cov <- cov(Xtrain_reduced)
Xtrain_pca<-prcomp(Xtrain_cov)
# tranforming the dataset/ applying PCA to normalized-raw data
Xtrain_final <-as.matrix(Xtrain_reduced) %*% Xtrain_pca$rotation[,1:60]


# Applying PCA to test data
#----------------------------
Xtest_reduced <- dftest.pixel/255
Xtest_final <-as.matrix(Xtest_reduced) %*% Xtrain_pca$rotation[,1:60]

#plots on the reduced data after principal component analysis
plot(Xtrain_pca$sdev)
plot(Xtrain_pca$x)
plot(Xtrain_pca$rotation)

# After seeing the above plots, the distribution of the data set is ellipsoid in nature and the hyper plan equation is of the form
# x^2/a^2 + y^2/b^2 + 1/c^2 = 1

#####################################################################################
# 5. Model Building 
#----------------------------
trainFinal <- cbind.data.frame(label, Xtrain_final)
names(trainFinal)[1] <- "label"
trainFinal$label <- as.factor(trainFinal$label)

testFinal <- cbind.data.frame(test_label, Xtest_final)
names(testFinal)[1] <- "label"
testFinal$label <- as.factor(testFinal$label)

# since the original train and test data contains 60000 and 10000 observations, for better performance
# of the model to run in the minimum time is better to prepare the model with the sample data

#since the data set is huge, take 30% of the data as sampling
indices_train = sample(1:nrow(trainFinal), 0.3*nrow(trainFinal))
indices_test = sample(1:nrow(testFinal), 0.3*nrow(testFinal))

trainFinal = trainFinal[indices_train,]
testFinal = trainFinal[indices_test,]

#--------------------------------------------------------------------
# 5.1 Linear model - SVM  at Cost(C) = 1
#####################################################################
# Model with C = 1
#----------------------------

# Start the clock!
ptm <- proc.time()
model_1<- ksvm(label ~ ., data = trainFinal,scale = FALSE, C=1)

# Predicting the model results 
evaluate_1<- predict(model_1, testFinal[,-1])

# Confusion Matrix - Finding accuracy, Sensitivity and specificity
conf_matrix_1 <- confusionMatrix(evaluate_1, testFinal$label, positive = "Yes")

print(conf_matrix_1)

#print the statistics of confusion matrix of model_1
if(is.defined(conf_matrix_1)){
  printConfusionMatrixStats(conf_matrix_1)
}
# Stop the clock
proc.time() - ptm
-------------------------------------------------------------------
#Statistics captured when run with the enire dataset i.e.  60000 observations of train data and 10000 observations of test data
# Accuracy                      - 0.979
# Sensitivity                   - 0.979 
# Specificity                   - 0.998
# processing time of the model  - 5.4 minutes
# Note: Here Sensitivity and Specificity are average because there are 10 classes and
#       each has its own sensitivity and specificity
  
#Statistics captured when run with the sample dataset i.e.  18000 observations of train data and 3000 observations of test data
# Accuracy                      - 0.982
# Sensitivity                   - 0.982 
# Specificity                   - 0.998
# processing time of the model  - 55.28 seconds
# Note: Here Sensitivity and Specificity are average because there are 10 classes and
#       each has its own sensitivity and specificity

#--------------------------------------------------------------------
# 5.2 Linear model - SVM  at Cost(C) = 10
#####################################################################
# Model with C = 10
#----------------------------

# Start the clock!
ptm <- proc.time()

model_2<- ksvm(label ~ ., data = trainFinal,scale = FALSE, C=10)

# Predicting the model results 
evaluate_2<- predict(model_2, testFinal[,-1])

# Confusion Matrix - Finding accuracy, Sensitivity and specificity
conf_matrix_2 <- confusionMatrix(evaluate_2, testFinal$label, positive = "Yes")

#print the statistics of confusion matrix of model_2
print(conf_matrix_2)
if(is.defined(conf_matrix_2)){
  printConfusionMatrixStats(conf_matrix_2)
}

# Stop the clock
proc.time() - ptm

#Statistics captured when run with the enire dataset i.e.  60000 observations of train data and 10000 observations of test data
# Accuracy    - 0.984
# Sensitivity - 0.984
# Specificity - 0.998
# Note: Here Sensitivity and Specificity are average because there are 10 classes and
#       each has its own sensitivity and specificity

#--------------------------------------------------------------------
# 5.3 Non Linear model - SVM
#####################################################################

# RBF kernel 
#----------------------------

# Start the clock!
ptm <- proc.time()
model_rbf <- ksvm(label ~ ., data =trainFinal,scale=FALSE, kernel = "rbfdot")


# Predicting the model results 
Eval_RBF<- predict(model_rbf, testFinal[,-1])

#confusion matrix - RBF Kernel
conf_matrix_model_rbf <- confusionMatrix(Eval_RBF,testFinal$label)
#print the statistics for confuusion matrix
print(conf_matrix_model_rbf)
printConfusionMatrixStats(conf_matrix_model_rbf)

# Stop the clock
proc.time() - ptm

# Accuracy    - 0.979
# Sensitivity - 0.979
# Specificity - 0.997

# Note: Here Sensitivity and Specificity are average because there are 10 classes and
#       each has its own sensitivity and specificity


#####################################################################################
# 6  Cross validation
#####################################################################
# 6.1 Hyperparameter tuning and Cross Validation - Linear - SVM 
#####################################################################

# We will use the train function from caret package to perform crossvalidation

# Start the clock!
ptm <- proc.time()

trainControl <- trainControl(method="cv", number=5)
# Number - Number of folds 
# Method - cross validation

metric <- "Accuracy"

set.seed(100)

# Making grid of  C values. 
grid <- expand.grid(C=seq(1, 5, by=1))

# Performing 5-fold cross validation

fit.svm <- train(label~., data=trainFinal, method="svmLinear", metric=metric, 
                        tuneGrid=grid, trControl=trainControl)

# Printing cross validation result
print(fit.svm)
# Best tune  C=1, Accuracy - 0.9296334

# Plotting model results
plot(fit.svm)

###############################################################################

# Valdiating the model after cross validation on test data

evaluate_linear_test<- predict(fit.svm, testFinal)
conf_matrix <- confusionMatrix(evaluate_linear_test, testFinal$label)


#print the statistics of confusion matrix 
print("Model using Hyperparameter tuning and Cross Validation - Linear - SVM ")
print(conf_matrix)
printConfusionMatrixStats(conf_matrix)

# Stop the clock
proc.time() - ptm

# Accuracy    - 0.9358
# Sensitivity - 0.934723
# Specificity - 0.992872

# Note: Here Sensitivity and Specificity are average because there are 10 classes and
#       each has its own sensitivity and specificity

#####################################################################
# 6.2. Hyperparameter tuning and Cross Validation - Non-Linear - SVM 
#####################################################################

# Start the clock!
ptm <- proc.time()

trainControl <- trainControl(method="cv", number=5)
# Number - Number of folds 
# Method - cross validation

metric <- "Accuracy"

set.seed(100)

# Making grid of "sigma" and C values. 
grid <- expand.grid(.sigma=seq(0.01, 0.05, by=0.01), .C=seq(1, 5, by=1))


# Performing 5-fold cross validation
fit.svm_radial <- train(label~., data=trainFinal, method="svmRadial", metric=metric, 
                        tuneGrid=grid, trControl=trainControl)

# Printing cross validation result
print(fit.svm_radial)
# Best tune at sigma = 0.03 & C=2, Accuracy - 0.985

# Plotting model results
plot(fit.svm_radial)


######################################################################
# Checking overfitting - Non-Linear - SVM
######################################################################

# Validating the model results on test data
evaluate_non_linear<- predict(fit.svm_radial, testFinal)
conf_matrix_non_linear <- confusionMatrix(evaluate_non_linear, testFinal$label)

print(conf_matrix_non_linear)
printConfusionMatrixStats(conf_matrix_non_linear)
# Stop the clock
proc.time() - ptm

# Accuracy    - 0.985
# Sensitivity - 0.984
# Specificity - 0.998
# Note: Here Sensitivity and Specificity are average because there are 10 classes and
#       each has its own sensitivity and specificity

######################################################################
# 7. Conclusion/Summary
######################################################################
#S.NO	"Model Name"	                     "Accuracy"	"Average Sensitivity"	"Average Specificity"	"processing time"	"C value"	    "E value"	"Remarks"
# 1	  linear model  with C = 1             0.979	      0.979	                0.998	               5.4  minutes	      	1	        	 NA	
# 2	  linear model  with C = 10 	         0.984	      0.984	                0.998	               3.2  minutes	      	10	      	     NA	
# 3	  Model Non-Linear - SVM	             0.979	      0.979	                0.997	               4.66 minutes	      	NA     	  	     NA	        
# 4	  Cross Validation - Linear - SVM	     0.936	      0.931	                0.941	               70   minutes	      	1 to 5	  	     NA	        Best tune  C=1, Accuracy - 0.9296334
# 5	  Cross Validation - Non-Linear - SVM  0.985	      0.984	                0.998	               20   hours	          1 to 5	  	0.1 to 0.5        The final values used for the model were sigma = 0.03 and C = 2	

#After compapring all the SVM models for both linear and Non linear followoing are the conclusions:
#--------------------------------------------------------------------------------------------------

#1. The best tune model is #5 'SVM Non-linear with sigma = 0.03 & C=2, Accuracy - 0.985', but if you see the  
#   procesing time, it takes 20 hour to arrive at this conclusion, also non-linearity is little high and little complex in terms of 
#   explaining it to the end user

#2. Also in all the models accuracy, sensitivity and specificy are almost the same with variation of 0.1
#   So lets choose the linear model which is comparatively less complex compared to non-linear model
#   So second best model in terms of accuracy is #2 'linear model  with C = 10' but it is overfitted.

#3  If you take processing time, the minimum time taken is 3.2 minutes and model is #2 "linear model  with C = 10"
#   but here cost of overfitting is high and it is 10

####################################------END OF ASSIGNMENT--------##################################


