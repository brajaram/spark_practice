from __future__ import print_function
import sys

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: home_work_1 <input file name>", file=sys.stderr)
        exit(-1)

    inputFileName = sys.argv[1]

    from pyspark import SparkContext, SparkConf

    conf = SparkConf().setAppName("HomeWorkOne")
    sc = SparkContext(conf=conf)

    #Part 1
    #Store raw data as RDD with each element of RDD representing an instance with comma delimited strings.
    msDataRDD = sc.textFile(inputFileName)

    #Count the number of data points we have
    print('Number of data points {0}'.format(msDataRDD.count()))
    #Print the list of first 40 instances
    print('list of first 40 instances')

    for line in msDataRDD.top(40):
        print(line)

    #Part 2
    from pyspark.mllib.regression import LabeledPoint

    def parseLabeledPoint(line):
        columnValues = line.split(',')
        label, features = columnValues[0], columnValues[1:]
        return LabeledPoint(label, features)

    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import numpy as np

    sampleRowCount = 40
    sampleFortyLines = msDataRDD.take(sampleRowCount)

    #Part 3
    sampleLabeledPoints = map(parseLabeledPoint, sampleFortyLines)
    limitedFeatureValues = map(lambda lp: lp.features.toArray()[0:2], sampleLabeledPoints)
    limitedFeatureValues

    def plotHeatMap(x,y):
        plt.close()
        fig, ax = plt.subplots()
        fig = plt.gcf()
        fig.set_size_inches(8, 6)
        ax.set_frame_on(False)
        ax.set_yticks(np.arange(y) + 0.5, minor=False)
        ax.set_xticks(np.arange(x) + 0.5, minor=False)
        ax.invert_yaxis()
        ax.xaxis.tick_top()
        ax.set_xticklabels(map(lambda x : 'Feature_' + str(x),np.arange(1, x+1)), minor=False)
        ax.set_yticklabels(list(np.arange(1, y+1)), minor=False)
        ax.set_ylabel('Sample Instances')
        ax.set_xlabel('Features')
        return fig,ax

    fig,ax = plotHeatMap(2,40)
    image = plt.imshow(limitedFeatureValues,interpolation='nearest', aspect='auto', cmap=cm.Greys)
    fig.colorbar(image)
    plt.show()

    #Part 4
    #In learning problem, its natural to shift labels if its not starting from zero.
    #Find out the range of prediction year and shift labels if necessary so that lowest one starts from zero.
    labels = msDataRDD.map(lambda x: x.split(',')[0]).collect()
    minYear = float(min(labels))

    print('Minimum Year {0}'.format(minYear))

    rawLabeledPoints = msDataRDD.map(parseLabeledPoint)
    labeledPoints = rawLabeledPoints.map(lambda lp: LabeledPoint(lp.label - minYear, lp.features))

    #Part 5
    #Split dataset into training, validation and test set.
    trainData, validationData, testData = labeledPoints.randomSplit([.7, .2, .1], 52)
    trainData.cache()
    testData.cache()
    trainCount = trainData.count()
    valCount = validationData.count()
    testCount = testData.count()

    print(trainCount, valCount, testCount, trainCount + valCount + testCount)
    print(labeledPoints.count())

    #Create a baseline model where we always provide the same prediction irrespective of our input. (Use training data)
    # Implement a function to give Root mean square error given a RDD.
    averageTrainYear = (trainData.map(lambda pt: pt.label).mean())

    print(averageTrainYear)

    def rootMeanSqrdError(targetAndPred):
        return np.sqrt(targetAndPred.map(lambda targetAndPredTuple: (targetAndPredTuple[0] - targetAndPredTuple[1]) ** 2 ).mean())

    targetAndPredsTrain = trainData.map(lambda points: (points.label, averageTrainYear))
    targetAndPredsVal = validationData.map(lambda points: (points.label, averageTrainYear))
    targetAndPredsTest = testData.map(lambda points: (points.label, averageTrainYear))
    rmseTestData = rootMeanSqrdError(targetAndPredsTest)

    #Measure our performance of base model using it. (Use test data)
    print('Baseline Root Main Squared Error of Test Data = {0:.3f}'.format(rmseTestData))

    #Part 6
    #Visualize predicted vs actual using a scatter plot.
    from matplotlib.colors import ListedColormap, Normalize
    from matplotlib.cm import get_cmap
    cmap = get_cmap('YlOrRd')
    norm = Normalize()

    actual = np.asarray(testData.map(lambda lp: lp.label).collect())
    predictions = np.asarray(testData.map(lambda lp: averageTrainYear).collect())
    error = np.asarray(testData
                       .map(lambda lp: (lp.label, averageTrainYear))
                       .map(lambda (l, p): squaredError(l, p))
                       .collect())

    colors = cmap(np.asarray(norm(error)))[:,0:3]

    #scatter plot
    plt.scatter(predictions, actual, s=15**2, c=colors,  alpha=0.70, linewidths=0.5)
    plt.show()
