import sys
import math
import collections
import os 

from utils import GetWordListsAndNumberOffiles
from utils import ReadFile
from utils import stop_words


if(len(sys.argv) == 3):
    train_path = sys.argv[1]
    test_path = sys.argv[2]    
else:
    sys.exit("Please give right number of arguments-TRAININING PATH  containing both ham and spam folder> \
                                                    TEST PATH containing both ham and spam folder>")
    
#We need to find 
#P(ham) = number of documents belonging to category ham / Total Number of documents
#P(spam) = number of documents belonging to category spam / Total Number of documents
#P(ham|bodyText) = (P(ham) * P(bodyText|ham)) / P(bodyText)
#P(bodyText|spam) = P(word1|spam) * P(word2|spam)*.....
#P(bodyText|ham) = P(word1|ham) * P(word2|ham)*.....
#P(word1|spam) = count of word1 belonging to category spam / Total count of words belonging to category spam 
#P(word1|ham) = count of word1 belonging to category ham / Total count of words belonging to category ham    
#For new word not seen yet in test document
#P(new-word|ham) or P(new-word|spam) = 0
#This will make the product zero so we can solve this
# if (log(P(ham|bodyText)) > log(P(spam|bodyText)))
#    return 'ham'
#else:
#    return 'spam'
#    
#log(P(ham|bodyText)) = log(P(ham)) + log(P(bodyText|ham))
#                     = log(P(ham)) + log(P(word1|ham)) + log(P(word2|ham)) + .....
    
#P(word1|ham) = (count of word1 belonging to category ham + 1)/
#              (total number of words belonging to ham + number of distinct words in training database)
#P(word1|spam) = (count of word1 belonging to category spam + 1)/
#                 (total number of words belonging to spam + number of distinct words in training database)    
#location of the folder for ham & spam for train and test

HamFolderPath = train_path + '/ham'
SpamFolderPath = train_path + '/spam'

#Find out all the words in ham folder and spam folder and find there counts  
NumberOfHam, NumberOfSpam = 0, 0
WordListInham = []
WordListInspam = []
WordListInham,NumberOfHam = GetWordListsAndNumberOffiles(HamFolderPath)
WordListInspam,NumberOfSpam = GetWordListsAndNumberOffiles(SpamFolderPath)

#Function to Find out P(ham) and P(spam), by calculating
#the number of ham/spam documents and total number of documents
def FindPHamOrSpam(HamOrSpam):
    if HamOrSpam == "spam":
        Pspam = NumberOfSpam/(NumberOfSpam + NumberOfHam)
        return Pspam
    else:
        Pham = NumberOfHam/(NumberOfSpam + NumberOfHam)
        return Pham

#After finding all the words in ham and spam files , we will find the distinct words and its count
HamDictionary = dict(collections.Counter(w.lower() for w in WordListInham))
SpamDictionary = dict(collections.Counter(w.lower() for w in WordListInspam))

#making bag of words for both ham and spam and further counting the count of each Distinct word in it
bagOfWords = WordListInham + WordListInspam
BagOfWordsDict = collections.Counter(bagOfWords)

def UpdateCountOfMissingWords(AllWords,HamSpamWords):
    for words in AllWords:
        if words not in HamSpamWords:
            HamSpamWords[words] = 0
            
#getting missing words in each Ham and Spam list and adding them and intializing their count= 0
UpdateCountOfMissingWords(BagOfWordsDict,HamDictionary)
UpdateCountOfMissingWords(BagOfWordsDict,SpamDictionary)

#P(word1|ham) = (count of word1 belonging to category ham + 1)/
#              (total number of words belonging to ham + number of distinct words in training database)
#P(word1|spam) = (count of word1 belonging to category spam + 1)/
#                 (total number of words belonging to spam + number of distinct words in training database)  
#Here, Counter contains total number of words belonging to ham/spam plus number of distinct words 
#in training dataset as we updated all the missing words in dictionary too
ProbOfHamWords = dict()
ProbOfSpamWords = dict()
def FindProbabilityOfWord(classifier,removestopwords):
    Counter = 0
    if(removestopwords ==1):
            for word in stop_words:
                if word in HamDictionary:
                    del HamDictionary[word]
                if word in SpamDictionary:
                    del SpamDictionary[word]
                if word in BagOfWordsDict:
                    del BagOfWordsDict[word]                    
    if classifier == "ham":
        for word in HamDictionary:
            Counter += (HamDictionary[word] + 1)
        for word in HamDictionary:
            ProbOfHamWords[word] = math.log((HamDictionary[word] + 1)/Counter ,2)
    elif classifier == "spam":
        for word in SpamDictionary:
            Counter += (SpamDictionary[word] + 1)
        for word in SpamDictionary:
            ProbOfSpamWords[word] = math.log((SpamDictionary[word] + 1)/Counter ,2) 
           
#caluculating probability for each word in ham and Spam folders 
FindProbabilityOfWord("ham",0)
FindProbabilityOfWord("spam",0) 


#Finally classify the emails as ham or spam    
def PredictHamOrSpam(pathToFile, classifier):
    ProbabilityOfHam = 0 
    ProbabilityOfSpam = 0 
    InCorrectlyClassified = 0
    NumberOfFiles = 0
                   
    if classifier == "spam":
        for fileName in os.listdir(pathToFile):
            words =ReadFile(fileName,pathToFile)
            #find actual P(ham) and P(spam) i.e. (number of ham documents / Total no of documents)
            ProbabilityOfHam = math.log(FindPHamOrSpam("ham"),2)
            ProbabilityOfSpam = math.log(FindPHamOrSpam("spam"),2)
            #log(P(ham|bodyText)) = log(P(ham)) + log(P(word1|ham)) + log(P(word2|ham)) + .... 
            for word in words:
                if word in ProbOfHamWords:
                    ProbabilityOfHam += ProbOfHamWords[word]
                if word in ProbOfSpamWords:
                    ProbabilityOfSpam += ProbOfSpamWords[word]
            NumberOfFiles +=1
            if(ProbabilityOfHam >= ProbabilityOfSpam):
                InCorrectlyClassified+=1
    if classifier == "ham":
        for fileName in os.listdir(pathToFile):
            words =ReadFile(fileName,pathToFile)
            #find actual P(ham) and P(spam) i.e. (number of ham documents / Total no of documents)
            ProbabilityOfHam = math.log(FindPHamOrSpam("ham"),2)
            ProbabilityOfSpam = math.log(FindPHamOrSpam("spam"),2)
            #log(P(ham|bodyText)) = log(P(ham)) + log(P(word1|ham)) + log(P(word2|ham)) + ....            
            for word in words:
                if word in ProbOfHamWords:
                    ProbabilityOfHam += ProbOfHamWords[word]
                if word in ProbOfSpamWords:
                    ProbabilityOfSpam += ProbOfSpamWords[word]
            NumberOfFiles +=1
            if(ProbabilityOfHam <= ProbabilityOfSpam):
                InCorrectlyClassified+=1
    return InCorrectlyClassified,NumberOfFiles 

print("Executing Naive Bayes with stop words for Ham & Spam test emails :")  

HamTestPath = test_path + '\ham'
SpamTestPath = test_path + '\spam'        
IncorrectlyClassifiedHam,TotalHamEmails = PredictHamOrSpam(HamTestPath, "ham")
IncorrectlyClassifiedSpam,TotalSpamEmails = PredictHamOrSpam(SpamTestPath,"spam")
AccuracyOfHamClassification = round(((TotalHamEmails - IncorrectlyClassifiedHam )/(TotalHamEmails ))*100,2)
AccuracyOfSpamClassification = round(((TotalSpamEmails -  IncorrectlyClassifiedSpam )/(TotalSpamEmails))*100,2)
AllEmailClassified = TotalHamEmails + TotalSpamEmails
TotalIncorrectClassified = IncorrectlyClassifiedHam + IncorrectlyClassifiedSpam
OverAllAccuracy = round(((AllEmailClassified  - TotalIncorrectClassified )/AllEmailClassified)*100,2)

print("\nTotal number of files: ", AllEmailClassified)
print("\nCalculating Accuracy over Ham Emails")
print("Total number of Ham Emails: ", TotalHamEmails)
print("Number of Emails Classified as Ham: ", TotalHamEmails - IncorrectlyClassifiedHam)
print("Number of Emails Classified as Spam: ",IncorrectlyClassifiedHam)
print("\nNaive Bayes Accuracy For Ham Emails Classification:" + str(AccuracyOfHamClassification) + "%")

print("\nCalculating Accuracy over Spam Emails")
print("Total number of Spam Emails: ", TotalSpamEmails)
print("Number of Emails Classified as Spam: ", TotalSpamEmails - IncorrectlyClassifiedSpam)
print("Number of Emails Classified as Ham: ",IncorrectlyClassifiedSpam)
print("\nNaive Bayes Accuracy For Spam Emails Classification: " + str(AccuracyOfSpamClassification) + "%") 

print("\nNaive Bayes Total accuracy for Test Emails: " + str(OverAllAccuracy) + "%")

print("\n")

print("Executing Naive Bayes after removing stop words")
FindProbabilityOfWord("ham",1)
FindProbabilityOfWord("spam",1) 

IncorrectlyClassifiedHam,TotalHamEmails = PredictHamOrSpam(HamTestPath, "ham")
IncorrectlyClassifiedSpam,TotalSpamEmails = PredictHamOrSpam(SpamTestPath,"spam")
AccuracyOfHamClassification = round(((TotalHamEmails - IncorrectlyClassifiedHam )/(TotalHamEmails ))*100,2)
AccuracyOfSpamClassification = round(((TotalSpamEmails -  IncorrectlyClassifiedSpam )/(TotalSpamEmails))*100,2)
AllEmailClassified = TotalHamEmails + TotalSpamEmails
TotalIncorrectClassified = IncorrectlyClassifiedHam + IncorrectlyClassifiedSpam
OverAllAccuracy = round(((AllEmailClassified  - TotalIncorrectClassified )/AllEmailClassified)*100,2)

print("\nTotal number of files: ", AllEmailClassified)
print("\nCalculating Accuracy over Ham Emails")
print("Total number of Ham Emails: ", TotalHamEmails)
print("Number of Emails Classified as Ham: ", TotalHamEmails - IncorrectlyClassifiedHam)
print("Number of Emails Classified as Spam: ",IncorrectlyClassifiedHam)
print("\nNaive Bayes Accuracy For Ham Emails Classification:" + str(AccuracyOfHamClassification) + "%")

print("\nCalculating Accuracy over Spam Emails")
print("Total number of Spam Emails: ", TotalSpamEmails)
print("Number of Emails Classified as Spam: ", TotalSpamEmails - IncorrectlyClassifiedSpam)
print("Number of Emails Classified as Ham: ",IncorrectlyClassifiedSpam)
print("\nNaive Bayes Accuracy For Spam Emails Classification: " + str(AccuracyOfSpamClassification) + "%") 

print("\nNaive Bayes Total accuracy for Test Emails: " + str(OverAllAccuracy) + "%")

print("\n")





    
    