
import sys
import collections
import os
import re
import codecs
import numpy


from utils import GetWordListsAndNumberOffiles
from utils import ReadFile
from utils import stop_words




if(len(sys.argv) == 6):
    train_path = sys.argv[1]
    test_path = sys.argv[2]
    RemoveStopWords = sys.argv[3]
    Lambda = float(sys.argv[4])
    Iteration = sys.argv[5]
    
else:
    sys.exit("Please provide right number of Arguments-\
             \n<TRAINING FOLDER path  \
              \n<TESTING FOLDER path \
              \n<Remove stop words yes or no\
              \n<Regularization parameter (Lambda value)>\
              \n<Number of Iteration>")

#location of the folder for ham & spam for trainining 
HamTrainPath = train_path + '/ham'
SpamTrainPath = train_path + '/spam'

#Find out all the words in trainging file of ham and spam folders respectively
#and create Listof all the words
WordListInTrainHam,TrainHamFileCount= GetWordListsAndNumberOffiles(HamTrainPath)
WordListInTrainSpam,TrainSpamFileCount = GetWordListsAndNumberOffiles(SpamTrainPath)

def FilterStopWords():
    for word in stop_words:
        if word in WordListInTrainHam:
            WordListInTrainHam.remove(word)
        if word in WordListInTrainSpam:
            WordListInTrainSpam.remove(word)

if(sys.argv[3] == "yes"):
    FilterStopWords()
    
#Combine all the words from Ham and Spam files and take total count of files
AllWords = WordListInTrainHam + WordListInTrainSpam
AllFiles = TrainHamFileCount + TrainSpamFileCount

#Remove duplicate words from all word list
#collections.Counter will create a dictionary with each word as key and its count as value
#After that we are making list of unique words 
UniqueWords_counter = collections.Counter(AllWords)
UniqueWords = list(UniqueWords_counter.keys())

#Vectorize each file and all the words
#here each word is a feature and rows are each file
#initialize each value to zero
def CreateFeatureVector(row, column):
    featureVector = [[0 for columnNumber in range(column)] for rowNumber in range(row)]
    return featureVector

#Create training feature vector
trainingFeatureVector = CreateFeatureVector(AllFiles,len(UniqueWords) )

#Create List of classifier which will contain type of file (ham or spam)
ClassifierList = []
sigmoidList = []

for i in range(AllFiles):
    sigmoidList.append(-1)
    ClassifierList.append(-1)

#Fill the feature Vector
rowMatrix = 0
def FillFeatureVector(featureVector,path,UniqueWords,rowMatrix,classifier,ClassifierList):
    for fileName in os.listdir(path):
        words = ReadFile(fileName,path)
        temp = dict(collections.Counter(words))
        for key in temp:
            if key in UniqueWords:
                column = UniqueWords.index(key)
                featureVector[rowMatrix][column] = temp[key]        
        if(classifier == "ham"):
            ClassifierList[rowMatrix] =0
        elif(classifier == "spam"):
            ClassifierList[rowMatrix] = 1
        rowMatrix +=1
    return featureVector,rowMatrix,ClassifierList
 

       
#train matrix including ham and spam
trainingFeatureVector,rowMatrix,ClassifierList= FillFeatureVector(trainingFeatureVector,
                                                           HamTrainPath,
                                                           UniqueWords,
                                                           rowMatrix,
                                                           "ham",
                                                           ClassifierList)

trainingFeatureVector,rowMatrix,ClassifierList= FillFeatureVector(trainingFeatureVector,
                                                           SpamTrainPath,
                                                           UniqueWords,
                                                           rowMatrix,
                                                           "spam",
                                                           ClassifierList)  

#Logistic Regression
#1/1+exp(-z)
#P(Y=1|X) = 1/1+exp(w0 + summation of product of all feature with its weights)
#Classification Rule
#P(Y=1|X)/P(Y=0|X) > 1 then assign 1 otherwise 0
#Learn the weights
#

#Create the vector of weight for all feature and initialize it to 0
Featureweights = []
Featureweights = [0.0 for eachWord in range(len(UniqueWords))]

#for feature in range(len(UniqueWords)):
#    Featureweights.append(0) 

bias = 0 
learningRate = 0.001
regularization = Lambda
  
# for each column
def _sigmoid(z):
    return 1 / (1 + numpy.exp(-z))

#Calculate for each file 
 #Calculating P(Y=1|X) = 1/1+exp(w0 + summation of product of all feature with its weights)
def PopulateSigmoid(AllFiles,numberOffeatures,featureVector):
    global sigmoidList
    for files in range(AllFiles):
        summation = 1.0       
        for features in range(numberOffeatures):
            summation +=featureVector[files][features] * Featureweights[features]
        sigmoidList[files] = _sigmoid(summation)

#Weight update rule
#new_weight = old_weight 
#              + learningRate * (Sum of all (feature * P(Y=1|X, w))) 
#              - learningRate * regularization * oldweight
#Here, P(Y=1|X, w) is calculated using sigmoid function

def UpdateWeights(AllFiles,numberOfFeature,featureVector,ClassifierList):
    global sigMoidList
    
    for feature in range(numberOfFeature):
        weight = bias
        for files in range(AllFiles):
            frequency = featureVector[files][feature]
            y = ClassifierList[files]
            sigmoidValue = sigmoidList[files]
            weight += frequency * (y - sigmoidValue)
        
        oldWeight = Featureweights[feature]
        # weight update formula as given on slide 26 of Logistic Regression.pdf
        Featureweights[feature] += ((weight * learningRate) - (learningRate * regularization * oldWeight ) ) 
    return Featureweights        
        
##Train and find out weights
def Training(AllFiles, numberOffeatures,trainingFeatureVector,ClassifierList):
    PopulateSigmoid(AllFiles, numberOffeatures,trainingFeatureVector)
    UpdateWeights(AllFiles, numberOffeatures,trainingFeatureVector,ClassifierList)  

print("Train Logistics algorithm - ")
for i in range(int(Iteration)):
    print(i, end = ' ')
    Training(AllFiles, len(UniqueWords),trainingFeatureVector,ClassifierList) 
    
print("Trained the weights successfully")


HamTestPath = test_path + '\ham'
SpamTestPath = test_path + '\spam'

#Find out all the words in trainging file of ham and spam folders respectively
#and create Listof all the words
WordListInTestHam,TestHamFileCount= GetWordListsAndNumberOffiles(HamTestPath)
WordListInTestSpam,TestSpamFileCount = GetWordListsAndNumberOffiles(SpamTestPath)

def FilterTestStopWords():
    for word in stop_words:
        if word in WordListInTestHam:
            WordListInTestHam.remove(word)
        if word in WordListInTestSpam:
            WordListInTestSpam.remove(word)

if(sys.argv[3] == "yes"):
    FilterTestStopWords()
    print("\n Removed stop words")
else:
    print("\n Stop words not removed")
    
#Combine all the words from Ham and Spam files and take total count of files
AllTestWords = WordListInTestHam + WordListInTestSpam
AllTestFiles = TestHamFileCount + TestSpamFileCount

#Remove duplicate words from all word list
#collections.Counter will create a dictionary with each word as key and its count as value
#After that we are making list of unique words 
UniqueTestWords_counter = collections.Counter(AllTestWords)
UniqueTestWords = list(UniqueTestWords_counter.keys())

#test matrix including ham and spam
testClassifierList = []
rowTestMatrix=0

for i in range(AllTestFiles):
    testClassifierList.append(-1)
    
    
testFeatureVector = CreateFeatureVector(AllTestFiles,len(UniqueTestWords) )

testFeatureVector,rowTestMatrix,testClassifierList= FillFeatureVector(testFeatureVector,
                                                           HamTestPath,
                                                           UniqueTestWords,
                                                           rowTestMatrix,
                                                           "ham",
                                                           testClassifierList)

testFeatureVector,rowTestMatrix,testClassifierList= FillFeatureVector(testFeatureVector,
                                                           SpamTestPath,
                                                           UniqueTestWords,
                                                           rowTestMatrix,
                                                           "spam",
                                                           testClassifierList)

## Classify the test email and find out accuracy
def ClassifyEmailAsHamOrSpam():    
    numberOfHamsCorrectlyClassified, numberOfHamsIncorrectlyClassified = 0, 0
    numberOfSpamsCorrectlyClassified, numberOfSpamIncorrectlyClassified = 0, 0  
    
    for file in range(AllFiles):
        summationOfWeightAndFrequency = 1.0
        for wordIndex in range(len(UniqueTestWords)):
            word= UniqueTestWords[wordIndex]
            
            if word in UniqueWords:
                indexOfWord = UniqueWords.index(word)
                weight= Featureweights[indexOfWord]
                wordcount = testFeatureVector[file][wordIndex]
                
                summationOfWeightAndFrequency += weight*wordcount
                
        try:
            valueOfSigmoidFunction = _sigmoid(summationOfWeightAndFrequency)
        except Exception:
            print("Exception while executing sigmoid function.")
            
        if(testClassifierList[file] == 0):
            if valueOfSigmoidFunction < 0.5:
                numberOfHamsCorrectlyClassified += 1.0
            else:
                numberOfHamsIncorrectlyClassified += 1.0
        else:
            if valueOfSigmoidFunction >= 0.5:
                numberOfSpamsCorrectlyClassified += 1.0
            else:
                numberOfSpamIncorrectlyClassified += 1.0
    
    accuracyOnHamFiles = round((numberOfHamsCorrectlyClassified * 100)/(numberOfHamsCorrectlyClassified + numberOfHamsIncorrectlyClassified) , 2)            
    accuracyOnSpamFiles = round((numberOfSpamsCorrectlyClassified * 100) / (numberOfSpamsCorrectlyClassified + numberOfSpamIncorrectlyClassified), 2)
    print("Accuracy over Ham files: {}%".format(str(accuracyOnHamFiles)))
    print("Accuracy over Spam files: {}%".format(str(accuracyOnSpamFiles)))        
    
print("Completed Training")
print("\nPlease wait while classifying the test emails..\n")
ClassifyEmailAsHamOrSpam()



