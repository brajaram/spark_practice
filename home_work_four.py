from pyspark.sql import SQLContext
import sys
import numpy as np

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: home_ work_four <file name>", file=sys.stderr)
        exit(-1)

    from pyspark import SparkContext, SparkConf
    conf = SparkConf().setAppName("HomeWorkFour")
    sc = SparkContext(conf=conf)
    sqlContext = SQLContext(sc)

    log4j = sc._jvm.org.apache.log4j
    log4j.LogManager.getRootLogger().setLevel(log4j.Level.ERROR)

    def transform_data(line):
        fields = line.split(',')
        if fields[0] == "" or fields[0] is None or fields[0] == "ID" or fields[0] == "X1":
            return
        avg_payment_delay = float(np.mean([abs(float(x)) for x in fields[6:12]]))
        avg_bill = abs(float(np.mean([float(x) for x in fields[12:18]])))
        avg_payment = float(np.mean([float(x) for x in fields[18:24]]))
        return(float(fields[2]), float(fields[3]),\
                float(fields[4]), float(fields[5]),avg_payment_delay,\
                avg_bill, avg_payment, float(fields[24]))

    baseFileRdd = sc.textFile(sys.argv[1])
    print(baseFileRdd.take(3))
    transformedRdd = baseFileRdd.map(transform_data).filter(lambda x: x)
    print(transformedRdd.count())
    print(transformedRdd.take(5))

    from pyspark.mllib.regression import LabeledPoint

    # Convert to labeled vector
    labeledPoints = transformedRdd.map(lambda x: LabeledPoint(x[7], x[0:7]))
    print(labeledPoints.take(2))

    from pyspark.mllib.linalg import Vectors,VectorUDT
    from pyspark.sql.types import StructType,StructField,DoubleType

    schema = StructType([
        StructField("label", DoubleType(), True),
        StructField("features", VectorUDT(), True)
    ])

    labeledPointsDF = labeledPoints.toDF(schema)
    print(labeledPointsDF.printSchema())

    from pyspark.ml.feature import StringIndexer, VectorIndexer

    labelIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel").fit(labeledPointsDF)
    featureIndexer =\
        VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(labeledPointsDF)

    (trainingData, testData) = labeledPointsDF.randomSplit([0.7, 0.3])
    print('Training data count:', trainingData.count())
    print('Test data count    :', testData.count())

    #RandomForestClassifier
    from pyspark.ml.classification import RandomForestClassifier
    from pyspark.ml.evaluation import MulticlassClassificationEvaluator

    rf = RandomForestClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures")

    from pyspark.ml import Pipeline
    pipeline = Pipeline(stages=[labelIndexer, featureIndexer, rf])
    # Train model.  This also runs the indexers.
    model = pipeline.fit(trainingData)
    # Make predictions.
    predictions = model.transform(testData)
    # Select example rows to display.
    predictions.select("prediction", "indexedLabel", "features").show(5)
    # Select (prediction, true label) and compute test error
    evaluator = MulticlassClassificationEvaluator(
        labelCol="indexedLabel", predictionCol="prediction", metricName="precision")
    accuracy = evaluator.evaluate(predictions)
    print("Test Error = %g" % (1.0 - accuracy))
    rfModel = model.stages[2]
    print(rfModel)  # summary only

    """
    Test Error = 0.203659
    RandomForestClassificationModel (uid=rfc_0a7b98e648b4) with 20 trees
    """

    #DecisionTreeClassifier
    from pyspark.ml.classification import DecisionTreeClassifier

    dt = DecisionTreeClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures")
    pipeline = Pipeline(stages=[labelIndexer, featureIndexer, dt])
    model = pipeline.fit(trainingData)
    predictions = model.transform(testData)
    predictions.select("prediction", "indexedLabel", "features").show(5)
    # Select (prediction, true label) and compute test error
    evaluator = MulticlassClassificationEvaluator(
        labelCol="indexedLabel", predictionCol="prediction", metricName="precision")
    accuracy = evaluator.evaluate(predictions)
    print("Test Error = %g " % (1.0 - accuracy))
    treeModel = model.stages[2]
    # summary only
    print(treeModel)

    """
    Test Error = 0.203548
    DecisionTreeClassificationModel (uid=DecisionTreeClassifier_4b959a2698b0580f28b1) of depth 5 with 63 nodes
    """

    #GBTClassifier
    from pyspark.ml.classification import GBTClassifier
    gbt = GBTClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures", maxIter=10)
    pipeline = Pipeline(stages=[labelIndexer, featureIndexer, gbt])
    model = pipeline.fit(trainingData)
    predictions = model.transform(testData)
    predictions.select("prediction", "indexedLabel", "features").show(5)
    evaluator = MulticlassClassificationEvaluator(
        labelCol="indexedLabel", predictionCol="prediction", metricName="precision")
    accuracy = evaluator.evaluate(predictions)
    print("Test Error = %g" % (1.0 - accuracy))

    gbtModel = model.stages[2]
    print(gbtModel)  # summary only

    """
    Test Error = 0.200554
    GBTClassificationModel (uid=GBTClassifier_46cc9f052e532ea7acc1) with 10 trees
    """

    #NaiveBayes classification
    from pyspark.mllib.classification import NaiveBayes, NaiveBayesModel
    training, test = labeledPoints.randomSplit([0.7, 0.3], seed=0)
    model = NaiveBayes.train(training, 1.0)
    predictionAndLabel = test.map(lambda p: (model.predict(p.features), p.label))
    accuracy = 1.0 * predictionAndLabel.filter(lambda x: x[0] == x[1]).count() / test.count()
    print("Test Error = %g" % (1.0 - accuracy))

    """
    Test Error = 0.457906
    """
