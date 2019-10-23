import sys
import collections
import os
import re
import codecs
import numpy


if (len(sys.argv) != 6):  
    sys.exit("Please give valid Arguments- \n<path to TRAIN FOLDER that has both ham and spam folder> \
              \n<path to TEST FOLDER that has both ham and spam folder>\
              \n<yes or no to remove stop words\
              \n<Regularization parameters>\
              \n<iteration>")
else:
    train = sys.argv[1]
    test = sys.argv[2]
    Stop = sys.argv[3]
    Lamda = float(sys.argv[4])
    Iteration = sys.argv[5]

ham = list()
spam = list()
countTrainHam = 0
countTrainSpam = 0
dictProbHam = dict()
dictProbSpam = dict()
learningRate = 0.001
regularization = Lamda

stopWords = ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
             "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being", "below",
             "between", "both", "but", "by", "can't", "cannot", "could", "couldn't", "did", "didn't",
             "do", "does", "doesn't", "doing", "don't", "down", "during", "each", "few", "for", "from",
             "further", "had", "hadn't", "has", "hasn't", "have", "haven't", "having", "he", "he'd",
             "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how",
             "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its",
             "itself", "let's", "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of",
             "off", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over",
             "own", "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some",
             "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there",
             "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through",
             "to", "too", "under", "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've",
             "were", "weren't", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who",
             "who's", "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd", "you'll",
             "you're", "you've", "your", "yours", "yourself", "yourselves"]



bias = 0
xnode = 1
directoryHam = train + '/ham'
directorySpam = train + '/spam'
testHam = test + '/ham'
testSpam = test + '/spam'

# Regualar expression to clean the data given in train ham and spam folder
regex = re.compile(r'[A-Za-z0-9\']')

def FileOpen(filename, path):
    fileHandler = codecs.open(path + "\\" + filename, 'rU',
                              'latin-1')  # codecs handles -> UnicodeDecodeError: 'charmap' codec can't decode byte 0x9d in position 1651: character maps to <undefined>
    words = [Findwords.lower() for Findwords in re.findall('[A-Za-z0-9\']+', fileHandler.read())]
    fileHandler.close()
    return words

def browseDirectory(path):
    wordList = list()
    fileCount = 0
    for files in os.listdir(path):
        if files.endswith(".txt"):
            wordList += FileOpen(files, path)
            fileCount += 1
    return wordList, fileCount

# iterating through train to get the list of ham words used to form combined bag of words
ham, countTrainHam = browseDirectory(directoryHam)
spam, countTrainSpam = browseDirectory(directorySpam)

# iterating through test to get the list of ham words used to form combined bag of words
hamTest, countTestHam = browseDirectory(testHam)
SpamTest, countTestSpam = browseDirectory(testSpam)

def removeStopWords():
    #print('Begin')

    for word in stopWords:
        #print(word)
        if word in ham:
            i = 0
            lengthh=len(ham)
            while (i < lengthh):
                #print(i)
                if (ham[i] == word):
                    ham.remove(word)
                    lengthh = lengthh - 1
                    continue
                i = i + 1
            #ham.remove(word)
        #print('Done Ham')
        if word in spam:
            i = 0
            lengths=len(spam)
            while (i < lengths):
                if (spam[i] == word):
                    spam.remove(word)
                    lengths = lengths - 1
                    continue
                i = i + 1
            #spam.remove(word)
        #print('Done Spam')
        if word in hamTest:
            i = 0
            lengthht=len(hamTest)
            while (i < lengthht):
                if (hamTest[i] == word):
                    hamTest.remove(word)
                    lengthht = lengthht - 1
                    continue
                i = i + 1
            #hamTest.remove(word)
        #print('Done HamTest')
        if word in SpamTest:
            i = 0
            lengthst=len(SpamTest)
            while (i < lengthst):
                if (SpamTest[i] == word):
                    SpamTest.remove(word)
                    lengthst = lengthst - 1
                    continue
                i = i + 1
        #print('Done SpamTest')
            #SpamTest.remove(word)
    #with open('rham.txt', 'w') as filehandle:
    #    for listitem in ham:
    #        filehandle.write('%s\n' % listitem)
    #with open('rspam.txt', 'w') as filehandle:
    #    for listitem in spam:
    #        filehandle.write('%s\n' % listitem)
    #with open('rhamtest.txt', 'w') as filehandle:
    #    for listitem in hamTest:
    #        filehandle.write('%s\n' % listitem)
    #with open('rspamTest.txt', 'w') as filehandle:
    #    for listitem in SpamTest:
    #        filehandle.write('%s\n' % listitem)
    #print('End')

if (sys.argv[3] == "yes"):
    removeStopWords()


# collections.Counter counts the number of occurence of memebers in list
rawHam = dict(collections.Counter(w.lower() for w in ham))
dictHam = dict((k, int(v)) for k, v in rawHam.items())
rawSpam = dict(collections.Counter(w.lower() for w in spam))
dictSpam = dict((k, int(v)) for k, v in rawSpam.items())

bagOfWords = ham + spam
dictBagOfWords = collections.Counter(bagOfWords)
listBagOfWords = list(dictBagOfWords.keys())
TargetList = list()  # final value of ham or spam, ham = 1 & spam = 0
totalFiles = countTrainHam + countTrainSpam

rawTestHam = dict(collections.Counter(w.lower() for w in hamTest))
dictTestHam = dict((k, int(v)) for k, v in rawTestHam.items())
rawTestSpam = dict(collections.Counter(w.lower() for w in SpamTest))
dictTestSpam = dict((k, int(v)) for k, v in rawTestSpam.items())

# correct it to testham/spam
testBagOfWords = hamTest + SpamTest
testDictBagOfWords = collections.Counter(testBagOfWords)
testListBagOfWords = list(testDictBagOfWords.keys())
testTargetList = list()  # final value of ham or spam, ham = 1 & spam = 0
totalTestFiles = countTestHam + countTestSpam

#with open('afterrt.txt', 'w') as filehandle:
#    for listitem in testListBagOfWords:
#        filehandle.write('%s\n' % listitem)


# initialize matrix to zero
# use list comprehension to create this matrix
def initiliazeMatrix(row, column):
    featureMatrix = [0] * row
    for i in range(row):
        featureMatrix[i] = [0] * column
    return featureMatrix

trainFeatureMatrix = initiliazeMatrix(totalFiles, len(listBagOfWords))
testFeatureMatrix = initiliazeMatrix(totalTestFiles, len(testListBagOfWords))

rowMatrix = 0
testRowMatrix = 0

sigMoidList = list()  # for each row
for i in range(totalFiles):
    sigMoidList.append(-1)
    TargetList.append(-1)

for i in range(totalTestFiles):
    testTargetList.append(-1)

weightOfFeature = list()

for feature in range(len(listBagOfWords)):
    weightOfFeature.append(0)  # initializinf weight = 0


def makeMatrix(featureMatrix, path, listBagOfWords, rowMatrix, classifier, TargetList):
    for fileName in os.listdir(path):
        words = FileOpen(fileName, path)
        temp = dict(collections.Counter(words))
        for key in temp:
            if key in listBagOfWords:
                column = listBagOfWords.index(key)
                featureMatrix[rowMatrix][column] = temp[key]
        if (classifier == "ham"):
            TargetList[rowMatrix] = 0
        elif (classifier == "spam"):
            TargetList[rowMatrix] = 1
        rowMatrix += 1
    return featureMatrix, rowMatrix, TargetList


# train matrix including ham and spam
trainFeatureMatrix, rowMatrix, TargetList = makeMatrix(trainFeatureMatrix, directoryHam, listBagOfWords, rowMatrix,
                                                       "ham", TargetList)
trainFeatureMatrix, rowMatrix, TargetList = makeMatrix(trainFeatureMatrix, directorySpam, listBagOfWords, rowMatrix,
                                                       "spam", TargetList)

testFeatureMatrix, testRowMatrix, testTargetList = makeMatrix(testFeatureMatrix, testHam, testListBagOfWords,
                                                              testRowMatrix, "ham", testTargetList)
testFeatureMatrix, testRowMatrix, testTargetList = makeMatrix(testFeatureMatrix, testSpam, testListBagOfWords,
                                                              testRowMatrix, "spam", testTargetList)


# for each column
def sigmoid(x):
    den = (1 + numpy.exp(-x))
    sigma = 1 / den
    return sigma


# Calculate for each file
def sigmoidFunction(totalFiles, totalFeatures, featureMatrix):
    global sigMoidList
    for files in range(totalFiles):
        summation = 1.0

        for features in range(totalFeatures):
            summation += featureMatrix[files][features] * weightOfFeature[features]
        sigMoidList[files] = sigmoid(summation)


def calculateWeightUpdate(totalFiles, numberOfFeature, featureMatrix, TargetList):
    global sigMoidList

    for feature in range(numberOfFeature):
        weight = bias
        for files in range(totalFiles):
            frequency = featureMatrix[files][feature]
            y = TargetList[files]
            sigmoidValue = sigMoidList[files]
            weight += frequency * (y - sigmoidValue)

        oldW = weightOfFeature[feature]
        weightOfFeature[feature] += ((weight * learningRate) - (learningRate * regularization * oldW))

    return weightOfFeature


def trainingFunction(totalFiles, numbeOffeatures, trainFeatureMatrix, TargetList):
    sigmoidFunction(totalFiles, numbeOffeatures, trainFeatureMatrix)
    calculateWeightUpdate(totalFiles, numbeOffeatures, trainFeatureMatrix, TargetList)


def classifyData():
    correctHam = 0
    incorrectHam = 0
    correctSpam = 0
    incorrectSpam = 0
    overallAccuracy=0
    idx=0
    for file in range(totalTestFiles):
        print('TestFile : '+str(idx+1))
        summation = 1.0
        for i in range(len(testListBagOfWords)):
            word = testListBagOfWords[i]

            if word in listBagOfWords:
                index = listBagOfWords.index(word)
                weight = weightOfFeature[index]
                wordcount = testFeatureMatrix[file][i]

                summation += weight * wordcount

        sigSum = sigmoid(summation)
        if (testTargetList[file] == 0):
            if sigSum < 0.5:
                correctHam += 1.0
            else:
                incorrectHam += 1.0
        else:
            if sigSum >= 0.5:
                correctSpam += 1.0
            else:
                incorrectSpam += 1.0
        idx += 1
    print("Accuracy on Ham:" + str((correctHam / (correctHam + incorrectHam)) * 100))
    print("Accuracy on Spam:" + str((correctSpam / (correctSpam + incorrectSpam)) * 100))
    print("Overall Accuracy :" + str(((correctHam+correctSpam) / (correctHam + incorrectHam+correctSpam + incorrectSpam)) * 100))


print("Training the algorithm - ")
for i in range(int(Iteration)):
    print(i, end=' ')
    trainingFunction(totalFiles, len(listBagOfWords), trainFeatureMatrix, TargetList)


print("Training completed successfully")
print("\nPlease wait while classifying the data..\nIt may take few minutes")
classifyData()


