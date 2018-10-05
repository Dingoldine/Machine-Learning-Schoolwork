import numpy , random , math 
from scipy.optimize import minimize 
import matplotlib.pyplot as plt



class SupportVectorMachine:


	def initialize():
		numpy.random.seed(100)

		classA = numpy.concatenate(
			(numpy.random.randn(10, 2) * 0.2 + [1.5, 0.5],
			numpy.random.randn(10, 2) * 0.2 + [-1.5, 0.5]))

		classB = numpy.random.randn(20, 2) * 0.2 + [0.0, -0.5]

		inputs = numpy.concatenate((classA, classB))

		targets = numpy.concatenate((
			numpy.ones(classA.shape[0]),
			-numpy.ones(classB.shape[0])))

		N = inputs.shape[0] 

		permute=list(range(N))

		random.shuffle(permute)
		inputs = inputs[permute, :]
		targets = targets[permute]

		start = numpy.zeros(N)


		Pmatrix = numpy.zeros((N,N))
		for i in range(1,N):
			for j in range(1,N):
				Pmatrix[i][j] = targets[i]*targets[j]*linearKernel(inputs[i],inputs[j])

		print(N)

		return inputs,targets, start, Pmatrix 

		#plotData(classA, classB)

	def plotData(A, B):
		plt.plot([p[0] for p in A], [p[1] for p in A], 'b.')
		plt.plot([p[0] for p in B], [p[1] for p in B], 'r.')

		plt.axis('equal')
		plt.savefig('datapoints.pdf')
		plt.show()

	def zerofun():
		pass


	def linearKernel(x, y):
		return numpy.dot(x.T, y)


	def polynomialKernel():
		pass


	def objective(alpha):
		pass

				

		#multiplication ector/matrix
		#numpy.dot

		#vector sum
		#numpy.sum



	def main():

		a = numpy.array([5, 4])
		b = numpy.array([1, 2])

		res = linearKernel(a, b)

		

		datapoints, targets, start, Pmatrix = initialize()

		#print(datapoints)
		#print(targets)
		#print(start)
		#print(Pmatrix)
		ret = minimize(objective(alpha), start, bounds=B, constraints=XC)
		alpha = ret['x']


if __name__ == "__main__":
	main()
