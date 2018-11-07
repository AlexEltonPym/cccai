import numpy as np
import csv
import sys
from sklearn import preprocessing
import matplotlib.pyplot as plt
import random
import time as sleepy

questionCount = 16
sampleCount = 125

names = ['u', 's0','s1','s2','s3','s4','s5','s6','s7','s8','s9','q0','q1','q2','q3','a0', 'qc']

user = "u"
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
 #('Everything', [enjoy, skills, prepare, time, conscious, new, unexpected, learnt, better, motivated], []),
 ('Flow', [enjoy, skills, prepare, time, conscious], []),
]

def main():
  dataset = loadDataset('mvp2std.csv')
  samples = splitDataset(dataset, tests[0], sampleCount)

  plotResults(samples[0])

def plotResults(data):
  print(data[0])
  enjoyment = [row[0] for row in data]
  skill = [row[1] for row in data]
  challenge = [row[2] for row in data]
  

  minE = min(enjoyment)
  maxE = max(enjoyment)
  minS = min(skill)
  maxS = max(skill)
  minC = min(challenge)
  maxC = max(challenge)

  print(minE, maxE)
  for i in range(len(enjoyment)):
    enjoyment[i] = (enjoyment[i]-minE)/(maxE-minE)
    skill[i] = (skill[i]-minS)/(maxS-minS)+random.random()/10
    challenge[i] = (challenge[i]-minC)/(maxC-minC)+random.random()/10
  print()
  print(enjoyment)

  plt.scatter(skill, challenge, s=15, c=enjoyment, cmap="viridis", alpha=1)
  plt.colorbar(alpha=1)
  
  #plt.show()
  plt.savefig("flowDiagram.svg")

def loadDataset(filename):
  with open(filename, 'r') as csvfile:
    reader = csv.DictReader(csvfile)

    dataset = np.zeros((sampleCount, 1 + questionCount)) #initialise dataset array

    i = 0
    for row in reader:
      datum = np.array(int(row['u'])) #create user like this: 0000100000

      questions = np.fromiter(map(float, list(row.values())[1:questionCount+1]), dtype=np.float) #get question responses
      datum = np.append(datum, questions) #add questions to sample array
      dataset[i] = datum #add sample to dataset
      print(row)
      i = i + 1
    return dataset

def splitDataset(dataset, test, splitPoint):
  testName, x, y = test

 # np.random.shuffle(dataset)

  trainingData = dataset[:splitPoint]
  validationData = dataset[splitPoint:]

  trainingX = trainingData[..., nums(x)]
  trainingY = trainingData[..., nums(y)]
  
  validationX = validationData[..., nums(x)]
  validationY = validationData[..., nums(y)]
  return (trainingX, trainingY, validationX, validationY)

def nums(inputList):
  ret = []

  for name in inputList:
    ret.append(names.index(name))
  return ret

main()