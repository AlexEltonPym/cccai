import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
import csv
import sys

studentCount = 33
questionCount = 16
sampleCount = 125

names = ['u%d' % i for i in range(studentCount)] + ['s0','s1','s2','s3','s4','s5','s6','s7','s8','s9','q0','q1','q2','q3','a0','qc']

user = "user"
enjoy = "s0"
skills = "s1"
prepare = "s2"
time = "s3"
conscious = "s4"
new = "s5"
unexpected = "s6"
learnt = "s7"
better = "s8"
motivated = "s9"
quiz0 = "q0"
quiz1 = "q1"
quiz2 = "q2"
quiz3 = "q3"
assignment = "a0"
combined = "qc"

tests = [
  ('Everything (but enjoyment) -> enjoyment', [user, skills, prepare, time, conscious, new, unexpected, learnt, better, motivated], [enjoy]),
  ('Flow -> enjoyment + motivation', [skills, prepare, time, conscious], [enjoy, motivated]),
  ('Epistemic curiosity -> enjoyment + motivation', [new, unexpected, learnt, better], [enjoy, motivated]),
  ('Everything (including motivated) -> motivated', [user, skills, prepare, time, conscious, new, unexpected, learnt, better, motivated], [motivated]),
  ('Quizzes -> assignment', [quiz0, quiz1, quiz2, quiz3], [assignment]),
  ('QC -> Quizzes', [combined], [quiz0, quiz1, quiz2, quiz3, assignment])
]

percents = [0.7]

def main():
  dataset = loadDataset('mvp2std.csv')

  for test in tests:
    for p in percents:
      splitPoint = int(p * sampleCount)

      samples = splitDataset(dataset, test, splitPoint)

      results = runRegressor(samples)
      printResults(results, samples[3], test[0])


def loadDataset(filename):
  with open(filename, 'r') as csvfile:
    reader = csv.DictReader(csvfile)

    dataset = np.zeros((sampleCount, studentCount + questionCount)) #initialise dataset array

    i = 0
    for row in reader:
      datum = np.zeros(studentCount) #create user like this: 0000100000
      datum[int(row['u'])] = 1

      questions = np.fromiter(map(float, list(row.values())[1:questionCount+1]), dtype=np.int) #get question responses
      datum = np.append(datum, questions) #add questions to sample array
      dataset[i] = datum #add sample to dataset
      i = i + 1
    return dataset

def splitDataset(dataset, test, splitPoint):
  testName, x, y = test

  dataset = preprocessing.scale(dataset)

  np.random.shuffle(dataset)

  trainingData = dataset[:splitPoint]
  validationData = dataset[splitPoint:]

  trainingX = trainingData[..., nums(x)]
  trainingY = trainingData[..., nums(y)]
  
  validationX = validationData[..., nums(x)]
  validationY = validationData[..., nums(y)]
  return (trainingX, trainingY, validationX, validationY)

def runRegressor(tup):
  trainingX, trainingY, validationX, validationY = tup
  regr = RandomForestRegressor()

  if len(trainingY[0]) == 1:
    regr.fit(trainingX, np.ravel(trainingY))
  else:
    regr.fit(trainingX, trainingY)

  prediction = regr.predict(validationX)
  score = regr.score(validationX, validationY)
  return prediction, score



def printResults(results, answers, testName):
  prediction, score = results
  mse = mean_squared_error(answers, prediction)

  print("Test: " + testName)
  print("Training samples: " + str(sampleCount - len(answers)))
  print("Validation samples: " + str(len(answers)))
  print("Coefficient of determination: " + str(score))
  print("Mean squared error: " + str(mse) + "\n")

  #print(str(mse), file=sys.stderr)


def nums(inputList):
  ret = []

  for name in inputList:
    if(name is 'user'):
      ret.extend(range(studentCount))
    else:
      ret.append(names.index(name))
  return ret

main()