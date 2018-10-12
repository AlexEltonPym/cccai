#get rows, into big array
#convert student number into array of 00000100000 and combine with same for challenges
#fit that tree against output
		#output is Y
		#only fit 90% and save 10% for testing the fit
#predict data not used


import numpy as np
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor


import csv

studentCount = 31
challengeCount = 3
questionCount = 10

splitPercent = 0.7


with open('mvp.csv', 'r') as csvfile:
	reader = csv.DictReader(csvfile)

	dataset = np.empty_like(dtype={'names':['q0','q1','q2','q3','q4','q5','q6','q7','q8','q9','u']})

	for row in reader:
		datum = []

		user = np.zeros(studentCount, dtype=np.int) #user 0000100000
		user[int(row['u'])] = 1
		datum.append(user)

		datum.extend(map(int, list(row.values())[1:11])) #question responses 23415

		dataset.append(np.array(datum))

	splitPoint = int(splitPercent * len(dataset))
	dataset.shuffle()

	trainingData = dataset[:splitPoint]
	validationData = dataset[splitPoint:]

  

	trainingX = [sample[:trainingSplit] for sample in dataset[:splitPoint]]
	trainingY = [sample[trainingSplit:] for sample in dataset[:splitPoint]]
	
	validationX = [sample[:trainingSplit] for sample in dataset[splitPoint:]]
	validationY = [sample[trainingSplit:] for sample in dataset[splitPoint:]]

	# regr = tree.DecisionTreeRegressor()
	# regr.fit(trainingX, trainingY)

	regr = RandomForestRegressor()
	regr.fit(trainingX, np.ravel(trainingY))

	prediction = regr.predict(validationX)
	score = regr.score(validationX, validationY)

	# print(prediction)
	# print(validationY)

	print(score)

	# print(list([int(pred) for pred in prediction]))
	# print(list([int(correct[0]) for correct in validationY]))
