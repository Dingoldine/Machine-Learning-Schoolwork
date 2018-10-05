import numpy , random , math 
from scipy.optimize import minimize 
import matplotlib.pyplot as plt



class SupportVectorMachine:

	b_value = 0;



	def __init__(self):
		self.datapoints, self.targets, self.start, self.Pmatrix, self.N, self.classA, self.classB = self.initialize()
		#self.datapoints, self.targets, self.start, self.Pmatrix, self.N, self.classA, self.classB = self.debug()


	def debug(self):
		classA = numpy.array([ 1.78238922, 0.40465357])


	def initialize(self):
		sigma = 0.2
		A_cluster1 = 10
		A_cluster2 = 10
		B_cluster = 20

		numpy.random.seed(50)

		classA = numpy.concatenate(
			(numpy.random.randn(A_cluster1, 2) * sigma + [1.5, 0.5],
			numpy.random.randn(A_cluster2, 2) * sigma + [-1.5, 0.5]))

		classB = numpy.random.randn(B_cluster, 2) * sigma + [0.0, -0.5]

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

		self.plotData(classA, classB, "TrainingData")

		Pmatrix = numpy.zeros((N,N))
		for i in range(N):
			for j in range(N):
				Pmatrix[i, j] = targets[i]*targets[j]*self.kernel(inputs[i],inputs[j])

		

		return inputs,targets, start, Pmatrix, N, classA ,classB


	def plotData(self, A, B, name):
		plt.plot([p[0] for p in A], [p[1] for p in A], 'b.')
		plt.plot([p[0] for p in B], [p[1] for p in B], 'r.')

		plt.axis('equal')
		plt.savefig(name + '.png')
		plt.show()

	def zerofun(self, alpha):
		# sum_constraint = 0
		# dotProduct = numpy.sum(numpy.dot(alpha, self.targets))

		# sum_constraint = sum_constraint - dotProduct
		# return  (sum_constraint)

		return numpy.sum(numpy.dot(alpha, self.targets))

	def kernel(self, x, y):
		return self.polynomialKernel(x, y)
		#return self.linealKernel(x, y)
		#return self.radialBasisKernel(x, y)

	def linealKernel(self, x, y):
		return numpy.dot(x.T, y) 

	def polynomialKernel(self, x, y):
		power = 3 

		return (numpy.dot(x.T, y) + 1) ** power

	def radialBasisKernel(self, x, y):
		sigma =1.15

		euclidean_distance = numpy.linalg.norm(x-y)
		# % This is equivalent to computing the kernel on every pair of examples
		return math.exp(-euclidean_distance/(2.*sigma**2))

		
	def objective(self, alpha):
		# O = numpy.array(numpy.zeros((self.N, self.N)))
		# alphaValues = 0.5 * numpy.sum(numpy.dot(alpha, self.Pmatrix)) - numpy.sum(alpha) 
		# for i in range(self.N):
		# 	for j in range(self.N):
		# 		O[i,j] = alpha[i] * alpha[j] * self.kernel(alpha[i], alpha[j])

		# obj = 0.5 * numpy.sum(O) - numpy.sum(alpha)
		val = 0 
		for i in range(self.N):
		 	for j in range(self.N):
		 		val += alpha[i] * alpha[j] * self.Pmatrix[i, j]
		obj = ((1/2) * val) - numpy.sum(alpha)
		
		return obj

	def calculateThreshold(self, support_vectors, corresponding_targets, non_zero_alphas):
		#b = numpy.sum(numpy.dot(non_zero_alphas, corresponding_targets))
		b = 0 
	
		for i in range(len(non_zero_alphas)):
	
			b += (non_zero_alphas[i] * corresponding_targets[i] * self.kernel(support_vectors[0], support_vectors[i]))

		b = b - corresponding_targets[0]

		print("b value" ,b)

		return b

	
	def indicator(self, x, y, non_zero_alphas, corresponding_targets, support_vectors):
		array = numpy.array([x, y])
		ind = 0
		for i in range(len(non_zero_alphas)):
			ind += (non_zero_alphas[i] * corresponding_targets[i] *self.kernel(array, support_vectors[i]))

		ind = ind - self.b_value

		print(ind)
		return ind

	def plotBoundry(self, support_vectors, corresponding_targets, non_zero_alphas):
		xgrid = numpy.linspace(-5, 5)
		ygrid = numpy.linspace(-4, 4)

		grid = numpy.array([[self.indicator(x, y, non_zero_alphas, corresponding_targets, support_vectors) for x in xgrid] for y in ygrid])

		plt.contour(xgrid, ygrid, grid, 
					(-1.0, 0.0, 1.0),
					colors=('red', 'black', 'blue'),
					linewidths=(1, 3, 1))

		plt.plot([p[0] for p in self.classA], [p[1] for p in self.classA], 'b.')
		plt.plot([p[0] for p in self.classB], [p[1] for p in self.classB], 'r.')

		plt.plot([p[0] for p in support_vectors], [p[1] for p in support_vectors], 'g.', marker='x')
		plt.axis('equal')
		plt.savefig('boundry.png')
		plt.show()

	def run(self):

		C = 60 #Slack Parameter
		
		ret = minimize(self.objective, self.start, bounds=[(0, C) for b in range(self.N)], 
		 	constraints={'type':'eq', 'fun': self.zerofun})
		
		alpha = ret['x']

		threshold = 10**-5 

		
		non_zero_alphas = [ i for i in alpha if i > threshold]

		print("Non zero alpha values: ")
		print(non_zero_alphas)
		print("\n")

		indices_bigger_than_threshold = numpy.where(alpha > threshold)[0]
		
		support_vectors = self.datapoints[indices_bigger_than_threshold]
		print("Support Vectors: ")
		print(support_vectors)
		print("\n")
		print("len supp :", len(support_vectors))
		print(self.N)
		print("\n")
		print("targets of support_vectors")
		corresponding_targets = self.targets[indices_bigger_than_threshold]
		print(corresponding_targets)
		print("\n")


		plt.plot([p[0] for p in support_vectors], [p[1] for p in support_vectors], 'g.')
		axes = plt.gca()
		axes.set_xlim([-5,5])
		axes.set_ylim([-4,4])
		plt.show()


		self.b_value = self.calculateThreshold(support_vectors, corresponding_targets, non_zero_alphas)


		self.plotBoundry(support_vectors, corresponding_targets, non_zero_alphas)




def main():

	SVM = SupportVectorMachine()
	SVM.run()


if __name__ == "__main__":
	main()
