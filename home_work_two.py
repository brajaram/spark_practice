from __future__ import print_function
import sys

from pyspark.mllib.linalg import DenseVector

#Part 1
def costFunction(weights,lp):
    """
    function that computes the value (wT
    x - y) x and test this function on two examples.
    """
    return (weights.dot(lp.features) - lp.label)\
     * lp.features

#example one
weightOne = DenseVector([4, 5, 6])
lpExampleOne = LabeledPoint(3.0, [6, 2, 1])
costOne = costFunction(weightOne, lpExampleOne)
print('Cost of first example is {0}'.format(costOne))

#example two
weightTwo = DenseVector([1.5, 2.2, 3.4])
lpExampleTwo = LabeledPoint(5.0, [3.4, 4.1, 2.5])
costTwo = costFunction(weightTwo, lpExampleTwo)
print('Cost of second example is {0}'.format(costTwo))

#Part 2
def labelAndPrediction(weights, observation):
    """
    Implement a function that takes in weight and LabeledPoint instance and returns a <label, prediction tuple>
    """
    return (observation.label, weights.dot(observation.features))

predictionExampleRdd = sc.parallelize([LabeledPoint(3.0, np.array([6,2,1])),
                                    LabeledPoint(5.0, np.array([3.4, 4.1, 2.5]))])
labelAndPredictionOutput = predictionExampleRdd.map(lambda lp: labelAndPrediction(weightOne, lp))
print(labelAndPredictionOutput.collect())


#Part 3
def gradientDescent(trainData, numIters):
    """
    Implement a gradient descent function for linear regression.
    Test this function on an example.
    """
    n = trainData.count()
    noOfFeatures = len(trainData.take(1)[0].features)
    theta = np.zeros(noOfFeatures)
    alpha = 1.0
    trainingRMSE = np.zeros(numIters)
    for i in range(numIters):
        labelsAndPredsTrain = trainData.map(lambda lp: labelAndPrediction(theta, lp))
        trainingRMSE[i] = rootMeanSqrdError(labelsAndPredsTrain)
        gradient = trainData.map(lambda lp: costFunction(theta, lp)).sum()
        temp = alpha / (n * np.sqrt(i+1))
        theta -= temp * gradient
    return theta, trainingRMSE

n = 5
noOfFeatures = 5
numIters = 5
gradientExample = (sc
               .parallelize(trainData.take(n))
               .map(lambda lp: LabeledPoint(lp.label, lp.features[0:noOfFeatures])))
print(gradientExample.take(1))
exampleWeights, exampleTrainingError = gradientDescent(gradientExample, numIters)
print(exampleWeights)

#Part 4
#Train our model on training data and evaluate the model based on validation set.
numIters = 100
trainWeights, trainingRMSE = gradientDescent(trainData, numIters)
valLabelsAndPreds = validationData.map(lambda lp: labelAndPrediction(trainWeights, lp))
valRMSE = rootMeanSqrdError(valLabelsAndPreds)

print('Validation RMSE:\n\tTraining = {0:.3f}\n\tValidation = {1:.3f}'.format(trainRMSE,
                                                                       valRMSE))

#Part 5
norm = Normalize()
clrs = cmap(np.asarray(norm(np.log(trainError))))[:,0:3]
plt.scatter(range(0, numIters), np.log(trainError), s=14**2, c=clrs, edgecolors='#888888', alpha=0.75)


#Part 6
testLabelsAndPreds = testData.map(lambda lp: labelAndPrediction(trainWeights, lp))
testRMSE = rootMeanSqrdError(testLabelsAndPreds)
print('Test RMSE:\n\tTest = {0:.3f}'.format(testRMSE))
