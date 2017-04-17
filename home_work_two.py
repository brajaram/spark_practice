from __future__ import print_function
import sys
import numpy as np


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage: home_work_1 <input file name>", file=sys.stderr)
        exit(-1)

    inputFileName = sys.argv[1]

    from pyspark import SparkContext, SparkConf

    conf = SparkConf().setAppName("HomeWorkTwo")
    sc = SparkContext(conf=conf)

    #Normalize the features for GD convergance
    msDataRDD = sc.textFile(inputFileName)

    def rootMeanSqrdError(targetAndPred):
        return np.sqrt(targetAndPred.map(lambda targetAndPredTuple: (targetAndPredTuple[0] - targetAndPredTuple[1]) ** 2 ).mean())

    from pyspark.mllib.regression import LabeledPoint

    def parseLabeledPoint(line):
        columnValues = line.split(',')
        label, features = columnValues[0], columnValues[1:]
        return LabeledPoint(label, features)

    labels = msDataRDD.map(lambda x: x.split(',')[0]).collect()
    minYear = float(min(labels))

    rawLabeledPoints = msDataRDD.map(parseLabeledPoint)
    labeledPoints = rawLabeledPoints.map(lambda lp: LabeledPoint(lp.label - minYear, lp.features))


    labels = labeledPoints.map(lambda x: x.label)
    features = labeledPoints.map(lambda x: x.features)

    from pyspark.mllib.feature import Normalizer

    normalizer = Normalizer()
    data = labels.zip(normalizer.transform(features))
    parsedData = data.map(lambda lp: LabeledPoint(lp[0],lp[1]))

    #Part 1
    def lossFunction(weights,lp):
        """
        function that computes the value (wT
        x - y) x and test this function on two examples.
        """
        return np.dot((weights.dot(lp.features) - lp.label) , lp.features)

    from pyspark.mllib.linalg import DenseVector

    #test example one
    weightOne = DenseVector([4, 5, 6])
    lpExampleOne = LabeledPoint(3.0, [6, 2, 1])
    costOne = lossFunction(weightOne, lpExampleOne)
    print('Loss of first example is {0}'.format(costOne))

    #test example two
    weightTwo = DenseVector([1.5, 2.2, 3.4])
    lpExampleTwo = LabeledPoint(5.0, [3.4, 4.1, 2.5])
    costTwo =  lossFunction(weightTwo, lpExampleTwo)
    print('Loss of second example is {0}'.format(costTwo))

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
        noFeatures = len(trainData.take(1)[0].features)
        theta = np.zeros(noFeatures)
        learnRate = 1.0
        # We will compute and store the training error after each iteration
        errorTrain = np.zeros(numIters)
        for i in range(numIters):
            print('Iteration# {0} completed'.format(i+1))
            labelsAndPredsTrain = trainData.map(lambda lp: labelAndPrediction(theta, lp))
            errorTrain[i] = rootMeanSqrdError(labelsAndPredsTrain)
            gradient = trainData.map(lambda lp: lossFunction(theta, lp)).sum()
            tempLR = learnRate / (n * np.sqrt(i+1))
            theta -= tempLR * gradient
        return theta, errorTrain

    #split dataset
    trainData, validationData, testData = parsedData.randomSplit([.7, .2, .1], 52)
    trainData.cache()

    #test
    n = 5
    noOfFeatures = 5
    gradientExample = (sc
                   .parallelize(trainData.take(n))
                   .map(lambda lp: LabeledPoint(lp.label, lp.features[0:noOfFeatures])))
    print(gradientExample.take(1))
    exampleWeights, exampleTrainingError = gradientDescent(gradientExample, 5)
    print(exampleWeights)

    gradientExample.map(lambda lp: labelAndPrediction(exampleWeights, lp)).collect()

    #Part 4
    #Train our model on training data and evaluate the model based on validation set.
    numIters = 50
    trainWeights, trainingRMSE = gradientDescent(trainData, numIters)
    trainLabelAndPred = trainData.map(lambda lp: labelAndPrediction(trainWeights, lp))
    trainRMSE = rootMeanSqrdError(trainLabelAndPred)
    valLabelsAndPreds = validationData.map(lambda lp: labelAndPrediction(trainWeights, lp))
    valRMSE = rootMeanSqrdError(valLabelsAndPreds)
    print('Validation RMSE:\n\tTraining = {0:.3f}\n\tValidation = {1:.3f}'.format(trainRMSE,
                                                                           valRMSE))
    #Validation RMSE:
    #	Training = 11.948
    #	Validation = 11.943

    #Part 5
    from matplotlib.cm import get_cmap
    from matplotlib.colors import ListedColormap, Normalize
    import matplotlib.pyplot as plt

    norm = Normalize()
    cmap = get_cmap('YlOrRd')
    clrs = cmap(np.asarray(norm(np.log(trainingRMSE))))[:,0:3]
    fig, ax = plt.subplots()
    plt.scatter(range(0, 50), np.log(trainingRMSE), s=14**2, c=clrs, edgecolors='#888888', alpha=0.75)
    ax.set_xlabel('Iteration'), ax.set_ylabel(r'$\log_e(trainingRMSE)$')

    #Part 6
    testLabelsAndPreds = testData.map(lambda lp: labelAndPrediction(trainWeights, lp))
    testRMSE = rootMeanSqrdError(testLabelsAndPreds)
    print('Test RMSE:\n\tTest = {0:.3f}'.format(testRMSE))

    #Validation RMSE:
    #	Test = 11.990
