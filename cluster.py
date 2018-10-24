import numpy as np
from sklearn.manifold import TSNE
import csv
import sys
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


questionCount = 15
sampleCount = 125
kclusters = 5

names = ['u', 's0','s1','s2','s3','s4','s5','s6','s7','s8','s9','q0','q1','q2','q3','a0']

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


tests = [
  ('Everything', [enjoy, skills, prepare, time, conscious, new, unexpected, learnt, better, motivated], []),
]



def main():
  dataset = loadDataset('mvp2.csv')

  samples = splitDataset(dataset, tests[0], sampleCount)

  results = runManifold(samples)
  printResults(results)

  plotResults(results)

def plotResults(results):
  embedment, clusters = results
  X = [x[0] for x in embedment]
  Y = [y[1] for y in embedment]
  
  area = 10  # 0 to 15 point radii
  plt.set_cmap('Accent')

  plt.scatter(X, Y, s=area, c=clusters)
  plt.savefig("clusterfigk"+str(kclusters)+".svg")



def loadDataset(filename):
  with open(filename, 'r') as csvfile:
    reader = csv.DictReader(csvfile)

    dataset = np.zeros((sampleCount, 1 + questionCount)) #initialise dataset array

    i = 0
    for row in reader:
      datum = np.array(int(row['u'])) #create user like this: 0000100000

      questions = np.fromiter(map(float, list(row.values())[1:16]), dtype=np.int) #get question responses
      datum = np.append(datum, questions) #add questions to sample array
      dataset[i] = datum #add sample to dataset
      i = i + 1
    return dataset

def splitDataset(dataset, test, splitPoint):
  testName, x, y = test

  np.random.shuffle(dataset)

  trainingData = dataset[:splitPoint]
  validationData = dataset[splitPoint:]

  trainingX = trainingData[..., nums(x)]
  trainingY = trainingData[..., nums(y)]
  
  validationX = validationData[..., nums(x)]
  validationY = validationData[..., nums(y)]
  return (trainingX, trainingY, validationX, validationY)

def runManifold(tup):
  trainingX, trainingY, validationX, validationY = tup

  scaledX = preprocessing.scale(trainingX)

  tsneEmbedment = TSNE(n_components=2).fit_transform(scaledX)
  kmeansClusters = KMeans(n_clusters=kclusters, random_state=0).fit(scaledX).labels_

  return (tsneEmbedment, kmeansClusters)


def printResults(results):
  embedment, clusters = results

  print("Manifold")
  print("Training samples: " + str(sampleCount))
#	print(embedment)


  #print(str(mse), file=sys.stderr)


def nums(inputList):
  ret = []

  for name in inputList:
    ret.append(names.index(name))
  return ret

main()