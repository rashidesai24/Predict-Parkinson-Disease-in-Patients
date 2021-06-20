#LOADING THE REQUIRED PACKAGES

#PACKAGES FOR THE REPORT:
library(rmarkdown) #converting R markdown documents into several formats
library(knitr) #a general-purpose package for dynamic report generation
library(kableExtra) #nice table generator

#PACKAGES FOR THE CODE:
library(tidyverse) #for data processing and analysis
library(caret) #for machine learning
library(randomcoloR) #to generate a discrete color palette
library(GGally) #for the parallel coordinates chart
library(ggcorrplot) #for plotting the correlation matrix
library(reshape2) #for melt
library(MLmetrics) #for computing F1-score
library(caTools) #for logistic regression
library(e1071) #for support vector machines
library(nnet) #for neural network
library(rpart) #for decision tree
library(gbm) #for gradient boosting machine
library(randomForest) #for random forest

#IMPORTING DATA INTO R

url <-"https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data"
#creating a temporary file to download data into
temp <- tempfile()
#downloading from the url
download.file(url,temp)
#looking at the first few lines of the file
read_lines(temp, n_max = 3)
#separator is comma, file contains header
#reading the file into an R object
parkinsons <- read_csv(temp)
#unlinking the temporary file
unlink(temp)

str(parkinsons)
# checking for NAs
sum(is.na(parkinsons)) # no NAs present in the data
parkinsons$status <- as.factor(parkinsons$status)

parkinsons <- parkinsons %>% select(-name)

#-------------------------------------------------------------------------------------

#PRE-PROCESSING
#removing the column containing names since I want to make a general predictor,
#which can be extended to all

#identifying correlated predictors 
# corr <- parkinsons %>% select(-status) %>% cor() %>% round(1)
#flagging predictors for removal with a cutoff of 0.85
# highlyCorr <- findCorrelation(corr, cutoff=0.85)
#removing the columns from parkinsons
# parkinsons <- parkinsons %>% select(-status) %>% select(-all_of(highlyCorr)) %>% cbind(status=parkinsons$status)
str(parkinsons)


#Subset the features to exclude the target variable Status
pD_sub <- parkinsons[, -which(names(parkinsons) == "status")]
str(pD_sub)

cor(pD_sub)
View(parkinsons)

#Subsetting feature sets for PCA
#Jitter and Shimmer variables
MDVP = subset(parkinsons, select = c(1:7, 9:10, 13))
Shimmer_APQ = subset(parkinsons, select =c(11:12))
pca_comb = as.data.frame(c(MDVP,Shimmer_APQ))

#Computing Covariance and Eigen values
S <- cov(pD_sub)
sum(diag(S))

#Find Eigen values
s.eigen <- eigen(S)
s.eigen

#PERFORMING PCA on the subsetted data.
pr_comp = prcomp(pca_comb, center = TRUE, scale. = TRUE)
# scale = a logical value indicating if variables should be scaled for unit variance before analysis
pr_comp

#SUMMARIZE
summary(pr_comp)

# EXTRA
library(factoextra)
res.pca <- prcomp(pca_comb, scale = TRUE)
fviz_eig(res.pca)
fviz_pca_ind(res.pca,
             col.ind = "cos2", # Color by the quality of representation
             gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
             repel = TRUE)     # Avoid text overlapping
fviz_pca_var(res.pca,
             col.var = "contrib", # Color by contributions to the PC
             gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
             repel = TRUE)     # Avoid text overlapping


#Plotting Principal components with the target labels
library(ggfortify)
pca.plot <- autoplot(pr_comp, data = parkinsons, colour = 'status')
pca.plot

#Extracting the principal components into a dataframe
scaling <- pr_comp$sdev[1:2] * sqrt(nrow(pD_sub))
pc1 <- rowSums(t(t(sweep(pD_sub, 2 ,
                         colMeans(pD_sub))) * s.eigen$vectors[,1] * -1) / scaling[1])
pc2 <- rowSums(t(t(sweep(pD_sub, 2 ,
                         colMeans(pD_sub))) * s.eigen$vectors[,2] ) / scaling[2])

#Pooling the PC components into dataframe
df_pca <- data.frame(pc1, pc2, parkinsons$status, parkinsons$NHR, parkinsons$HNR, parkinsons$spread1,
                     parkinsons$spread2, parkinsons$D2, parkinsons$RPDE, parkinsons$DFA, parkinsons$PPE)
names(df_pca)[3] <- "status"

head(df_pca)

df_pca$pData_new.status = NULL

ggplot(df_pca, aes(x=pc1, y=pc2, color=parkinsons$status)) + 
  geom_point()

#checking the distribution of all predictors
df_pca %>% select(-status) %>% gather() %>%
  group_by(key) %>% mutate(mean=mean(value)) %>% ungroup() %>%
  ggplot(aes(value, y=..count.., fill=key))+
  geom_density() +
  geom_vline(aes(xintercept=mean)) + 
  facet_wrap(.~key, scales="free", ncol=4) +
  scale_fill_manual(values=distinctColorPalette(22)) +
  theme(legend.position = "none") +
  labs(x="Predictor", y="Count",
       title="Distribution of Predictors")

#-------------------------------------------------------------------------------------------------------------------------------------------

#Training data break up
set.seed(123)
train_size = floor(0.80*nrow(df_pca))
train_ind = sample(seq_len(nrow(df_pca)),size = train_size)

train = df_pca[train_ind,] #creates the training dataset with row numbers stored in train_ind
test = df_pca[-train_ind,]
View(train)

#---------------------------------------------------------------------------------------------------------------------

# Some visualizations to aid understanding of the distribution and features of the data

#setting the theme
theme_set(theme_minimal())

#checking the distribution of the outcome 'status'
train %>% mutate(status=factor(status, labels=c("Healthy","Parkinson's"))) %>% 
  ggplot(aes(status, fill=status)) +
  geom_bar() + 
  scale_fill_manual(values=c("mediumseagreen", "palevioletred")) +
  theme(legend.position = "none") +
  labs(x="Outcome (status)", y="Count",
       title="Distribution of Outcome")
#imbalanced classification therefore use kappa instead of accuracy as the metric

#checking the distribution of all predictors
train %>% select(-status) %>% gather() %>%
  group_by(key) %>% mutate(mean=mean(value)) %>% ungroup() %>%
  ggplot(aes(value, y=..count.., fill=key))+
  geom_density() +
  geom_vline(aes(xintercept=mean)) + 
  facet_wrap(.~key, scales="free", ncol=4) +
  scale_fill_manual(values=distinctColorPalette(10)) +
  theme(legend.position = "none") +
  labs(x="Predictor", y="Count",
       title="Distribution of Predictors")

#boxplots with jitter
train %>% mutate(status=factor(status, labels=c("Healthy","Parkinson's"))) %>%
  melt(id.vars="status") %>%
  ggplot(aes(status, value, fill=status)) + 
  facet_wrap(.~variable, scales="free", ncol=4) +
  geom_boxplot() + geom_jitter(color="grey", alpha=0.5) +
  theme(legend.position = "none") +
  scale_fill_manual(values=c("mediumseagreen", "palevioletred")) +
  labs(x="Outcome (status)", y="Value",
       title="Boxplots of Predictors vs. Status")

#-----------------------------------------------------------------------------------------------------------------------
#MACHINE LEARNING MODELS
#-------------------------------------------------------------------------------------------------------------------------------------

#outcome is 'status', 10 features available to use
#will use the confusion matrix as a metric along with accuracy
#false negatives can give false assurances, and false positives can send them to get needless, expensive medical tests
#positive class for confusionmatrix should be "1" since in medical science
#will use kappa as the metric in train functions since the classes of our outcome are imbalanced

#creating a copy of the train with status as a factor variable for use in ML algorithms
train_fct <- train %>% mutate(status=as.factor(status))

#setting the standard for 10-fold cross validation will be done with each algorithm
#because seeds need to be set according to tuning parameters for reproducible values

#-------------------------------------------------------------------------------------------------------------------------------------
#LOGISTIC REGRESSION
#-------------------------------------------------------------------------------------------------------------------------------------

#setting all the seeds for cross validation to get reproducible numbers
set.seed(1234)
seeds <- vector(mode = "list", length = 101)
for(i in 1:100) seeds[[i]] <- sample.int(1000, 1)
seeds[[101]] <- sample.int(1000, 1)
control <- trainControl(method = "repeatedcv", number = 10, repeats = 10, seeds=seeds)

#training the model
set.seed(1234)
train_logireg <- train(status ~ ., method = "LogitBoost", 
                       data = train_fct,
                       trControl = control,
                       metric="Kappa",
                       preProcess = c("center", "scale"),
                       tuneGrid = data.frame(nIter = seq(1, 400, 20)))
train_logireg
#plotting the parameters that were tuned
ggplot(train_logireg, highlight = TRUE) + labs(title="Tuning the Logistic Regression")
#storing the predicted values
predicted_logireg <- predict(train_logireg, test)
#computing metrics to assess efficacy of the algorithm
cm_logireg <- confusionMatrix(predicted_logireg, as.factor(test$status), positive="1")
cm_logireg
kappa_logireg <- cm_logireg$overall["Kappa"]
accu_logireg <- cm_logireg$overall["Accuracy"]
f1_logireg <- F1_Score(predicted_logireg, as.factor(test$status), positive="1")

l1 <- glm(status~., 
          data=train_fct, family="binomial")
summary(l1)
exp(coef(l1))
#-------------------------------------------------------------------------------------------------------------------------------------
#K-NEAREST NEIGHBOURS
#-------------------------------------------------------------------------------------------------------------------------------------

#setting all the seeds for cross validation to get reproducible numbers
set.seed(1234)
seeds <- vector(mode = "list", length = 101)
for(i in 1:100) seeds[[i]] <- sample.int(1000, 20)
seeds[[101]] <- sample.int(1000, 1)
control <- trainControl(method = "repeatedcv", number = 10, repeats = 10, seeds=seeds)

#training the model
set.seed(1234)
train_knn <- train(status ~ ., method = "knn", 
                   data = train_fct,
                   trControl = control,
                   metric="Kappa",
                   tuneGrid = data.frame(k = seq(1, 20, 1)),
                   preProcess = c("center", "scale"))
train_knn
#plotting the parameters that were tuned
ggplot(train_knn, highlight = TRUE) + labs(title="Tuning the K-Nearest Neighbours")
#storing the predicted values
predicted_knn <- predict(train_knn, test)
#computing metrics to assess efficacy of the algorithm
cm_knn <- confusionMatrix(predicted_knn, as.factor(test$status), positive="1")
cm_knn
kappa_knn <- cm_knn$overall["Kappa"]
accu_knn <- cm_knn$overall["Accuracy"]
f1_knn <- F1_Score(predicted_knn, as.factor(test$status), positive="1")

#-------------------------------------------------------------------------------------------------------------------------------------
#SUPPORT VECTOR MACHINE
#-------------------------------------------------------------------------------------------------------------------------------------

#setting all the seeds for cross validation to get reproducible numbers
set.seed(1234)
seeds <- vector(mode = "list", length = 101)
for(i in 1:100) seeds[[i]] <- sample.int(1000, 15)
seeds[[101]] <- sample.int(1000, 1)
control <- trainControl(method = "repeatedcv", number = 10, repeats = 10, seeds=seeds)

#training the model
set.seed(1234)
train_svm <- train(status ~ ., method = "svmLinearWeights", 
                   data = train_fct,
                   trControl = control,
                   metric="Kappa",
                   preProcess = c("center", "scale"),
                   tuneGrid = expand.grid(cost = c(.25, .5, 1), weight = c(1:5)))
train_svm
#plotting the parameters that were tuned
ggplot(train_svm, highlight = TRUE) + labs(title="Tuning the Support Vector Machine")
#storing the predicted values
predicted_svm <- predict(train_svm, test)
#computing metrics to assess efficacy of the algorithm
cm_svm <- confusionMatrix(predicted_svm, as.factor(test$status), positive="1")
cm_svm
kappa_svm <- cm_svm$overall["Kappa"]
accu_svm <- cm_svm$overall["Accuracy"]
f1_svm <- F1_Score(predicted_svm, as.factor(test$status), positive="1")


#-------------------------------------------------------------------------------------------------------------------------------------
#DECISION TREE
#-------------------------------------------------------------------------------------------------------------------------------------

#setting all the seeds for cross validation to get reproducible numbers
set.seed(1234)
seeds <- vector(mode = "list", length = 101)
for(i in 1:100) seeds[[i]] <- sample.int(1000, 1)
seeds[[101]] <- sample.int(1000, 1)
control <- trainControl(method = "repeatedcv", number = 10, repeats = 10, seeds=seeds)

#training the model
set.seed(1234)
train_rpart <- train(status ~ ., method = "rpart", 
                     data = train_fct,
                     metric="Kappa",
                     trControl = control,
                     tuneGrid = data.frame(cp = seq(0, 1, len = 25)))
train_rpart
#plotting the parameters that were tuned
ggplot(train_rpart, highlight = TRUE) + labs(title="Tuning the Decision Tree")
#storing the predicted values
predicted_rpart <- predict(train_rpart, test)
#computing metrics to assess efficacy of the algorithm
cm_rpart <- confusionMatrix(predicted_rpart, as.factor(test$status), positive="1")
cm_rpart
kappa_rpart <- cm_rpart$overall["Kappa"]
accu_rpart <- cm_rpart$overall["Accuracy"]
f1_rpart <- F1_Score(predicted_rpart, as.factor(test$status), positive="1")


# ENsemble

#ENSEMBLE (USING OUR TUNED PREDICTIONS)
pred_all <- data.frame(predicted_logireg, predicted_knn, predicted_svm, 
                       predicted_rpart)
predicted_esmb <- as.factor(ifelse(rowMeans(pred_all=="0")>0.5, 0, 1))
cm_esmb <- confusionMatrix(predicted_esmb, as.factor(test$status), positive="1")
cm_esmb
kappa_esmb <- cm_esmb$overall["Kappa"]
accu_esmb <- cm_esmb$overall["Accuracy"]
f1_esmb <- F1_Score(predicted_esmb, as.factor(test$status), positive="1")

results <- data.frame(model=c("Logistic Regression", "K Nearest Neighbours",
                              "Support Vector Machine",
                              "Decision Tree", "Ensemble"),
                      kappa=c(kappa_logireg, kappa_knn, kappa_svm,
                              kappa_rpart, kappa_esmb),
                      accuracy=c(accu_logireg, accu_knn, accu_svm,
                                 accu_rpart, accu_esmb),
                      F1_Score=c(f1_logireg, f1_knn, f1_svm, 
                                 f1_rpart, f1_esmb))
results
results %>% mutate(model=reorder(model,kappa)) %>%
  ggplot(aes(model, kappa)) + geom_col(width=0.5, fill="steelblue") + 
  coord_flip() +
  labs(x="Kappa", y="Model",
       title="Models' Performance Summary: Kappa")
results %>% mutate(model=reorder(model,accuracy)) %>%
  ggplot(aes(model, accuracy)) + geom_col(width=0.5, fill="goldenrod3") + 
  coord_flip() +
  labs(x="Accuracy", y="Model",
       title="Models' Performance Summary: Accuracy")
results %>% mutate(model=reorder(model,F1_Score)) %>%
  ggplot(aes(model, F1_Score)) + geom_col(width=0.5, fill="deeppink4") + 
  coord_flip() +
  labs(x="F1 Score", y="Model",
       title="Models' Performance Summary: F1 Score")
