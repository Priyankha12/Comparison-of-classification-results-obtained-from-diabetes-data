start_time <- Sys.time()
library(caTools)
library(rattle)
library(RWeka)
library(caret)
set.seed(123)
data <- read.csv("C:\\Users\\Dell\\Desktop\\Final year project\\CODE\\dataset\\diabetes_paper_median.csv")
Y=data[,9]
spl=sample.split(Y,SplitRatio = 0.8)
dataTrain = subset(data, spl==TRUE)
dataTest = subset(data, spl==FALSE)
data$Outcome=factor(data$Outcome,levels = c(0,1),labels = c("zero","one"))
resultJ48 <- J48(as.factor(Outcome)~., dataTrain, control=Weka_control(M=20,U=FALSE))
dataTest.pred <- predict(resultJ48, newdata = dataTest)
summary(resultJ48)
print(confusionMatrix(dataTest.pred,dataTest$Outcome,positive = "1"))
print(resultJ48)
end_time <- Sys.time()
print(end_time-start_time)
write_to_dot(resultJ48,"dotfilepaper.dot")
#system("dot -Tpng dotfile.dot -o c4.5_median.png")