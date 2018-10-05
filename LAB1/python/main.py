import monkdata as data
import dtree as tree 
import drawtree_qt5 as draw
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tkinter



def pruneTree(dataset, testSet):
	
	fractions = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
	errorList = []

	for x in fractions:
		train, val = partition(dataset, x)
		theTree = tree.buildTree(train, data.attributes)

		list_of_trees = tree.allPruned(theTree)


		theBest = 1000
		bestTree = 0

		for t in list_of_trees:
			error = 1 - tree.check(t, val)

			if error < theBest:
				theBest = error
				bestTree = t
		draw.drawTree(bestTree)
		smallest_error_at_fraction = 1 - tree.check(bestTree, testSet)
		errorList.append(smallest_error_at_fraction)

		# print ("smalest error")
		# print (smallest_error_at_fraction)
		# print ("occured at fraction")
		# print (x)

	return errorList

def partition(data, fraction): 
	ldata = list(data) 
	random.shuffle(ldata) 
	breakPoint = int(len(ldata) * fraction) 
	return ldata[:breakPoint], ldata[breakPoint:] 

def informationGain(data):
	a1 = tree.averageGain(data.monk3, data.attributes[0])
	a2 = tree.averageGain(data.monk3, data.attributes[1])
	a3 = tree.averageGain(data.monk3, data.attributes[2])
	a4 = tree.averageGain(data.monk3, data.attributes[3])
	a5 = tree.averageGain(data.monk3, data.attributes[4])
	a6 = tree.averageGain(data.monk3, data.attributes[5])

	# print(a1)

	# print(a2)

	# print(a3)

	# print(a4)

	# print(a5)

	# print(a6)

def main():
	print ("Entropy monk1")
	entropy1 = tree.entropy(data.monk1)
	print (entropy1)
	print ("\n")

	print ("Entropy monk2")
	entropy2 = tree.entropy(data.monk2)
	print (entropy2)
	print ("\n")

	print ("Entropy monk3")
	entropy3 = tree.entropy(data.monk3)
	print (entropy3)
	print ("\n")

	informationGain(data)

	#COMPUTING ENTROPY FOR SUBSET, WhY 0?!
	monk1Tree = tree.buildTree(data.monk1, data.attributes)
	#draw.drawTree(monk1Tree)
	#print(tree.bestAttribute(data.monk3, data.attributes))
	subSet = tree.select(data.monk1, data.attributes[4], 1)

	# newEntropy = tree.entropy(subSet)
	# print ("SubSet")
	# print (newEntropy)
	#END

	n = 0
	sumList = np.array([0.0] * 6)
	l1 = []
	l2 = []
	l3 = []
	l4 = []
	l5 = []
	l6 = []

	for x in range(100):
		errorList = np.array(pruneTree(data.monk1, data.monk1test))
		sumList += errorList
		l1.append(errorList[0])
		l2.append(errorList[1])
		l3.append(errorList[2])
		l4.append(errorList[3])
		l5.append(errorList[4])
		l6.append(errorList[5])

	finalList = sumList/100
	stdDevList = [np.std(l1),np.std(l2),np.std(l3),np.std(l4), np.std(l5),np.std(l6)]  

	print(finalList)
	print(stdDevList)

	line1, = plt.plot(finalList, label="Monk1 means", marker='o')
	# Create a legend for the first line.
	first_legend = plt.legend(handles=[line1], loc=1)

	x = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
	# create an index for each tick position
	xi = [i for i in range(0, len(x))]

	plt.xticks(xi, x)
	plt.ylabel('Mean Errors')
	plt.xlabel('Fractions')
	plt.show()


if __name__ == "__main__":
	main()
