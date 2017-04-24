from pyspark.sql import SparkSession
from pyspark.sql import Row
import math
from pyspark.ml.linalg import Vectors
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.classification import NaiveBayes, NaiveBayesModel
import numpy as np

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: home_work_three <google csv file name> <amazon csv file name>", file=sys.stderr)
        exit(-1)

    from pyspark import SparkContext, SparkConf
    conf = SparkConf().setAppName("HomeWorkFour")
    sc = SparkContext(conf=conf)

def transform_data(line):
    fields = line.split(',')
    if fields[0] == "" or fields[0] is None or fields[0] == "ID" fields[0] == "X1":
        return
    avg_payment_delay = np.mean([(abs(float(x)) for x in fields[6:12]])
    avg_bill = np.mean([(float(x) for x in fields[12:18]])
    avg_payment = np.mean([(float(x) for x in fields[18:24]])
    return (fields[0], float(fields[1]), float(fields[2]), float(fields[3]),\
            float(fields[4]), float(fields[5]),float(avg_payment_delay),\
            abs(float(avg_bill)), float(avg_payment), float(fields[24]))

file_name = "./data/default_of_credit_card_clients.csv"
baseFileRdd = sparkContext.textFile(file_name)
print baseFileRdd.take(3)
transformedRdd = baseFileRdd.map(transform_data)
print transformedRdd.count()
print transformedRdd.take(5)
# Convert to labeled vector
credit_labeled = credit.map(lambda l: (l[9], Vectors.dense(l[2:9])))
print credit_labeled.take(2)
credit_labeled_df = spark.createDataFrame(credit_labeled, ["label", "features"])
credit_labeled_df.show(5, False)

# Split into training and testing data
(training_data, test_data) = credit_labeled_df.randomSplit([0.7, 0.3])
print 'Training data count:', training_data.count()
print 'Test data count    :', test_data.count()

# Create the model
rf = RandomForestClassifier(labelCol="label", featuresCol="features")
rf_model = rf.fit(training_data)

# Model evaluator
evaluator = MulticlassClassificationEvaluator(predictionCol="prediction", labelCol="label", metricName="accuracy")

# Predict on the test data
predictions = rf_model.transform(test_data)
result = predictions.select("label", "features", "prediction")
result.where('label = 1.0').show(20, False)
result.where('label = 0.0').show(20, False)
print 'Random Forest accuracy : ', evaluator.evaluate(predictions)

'''
+-----+-----------------------------------------------------------------------------+----------+
|label|features                                                                     |prediction|
+-----+-----------------------------------------------------------------------------+----------+
|0.0  |[1.0,1.0,1.0,31.0,-1.3333333333333333,3382.0,7293.5]                         |0.0       |
|0.0  |[1.0,1.0,1.0,31.0,-1.3333333333333333,9791.833333333334,7911.5]              |0.0       |
|0.0  |[1.0,1.0,1.0,31.0,1.3333333333333333,54426.333333333336,2635.5]              |1.0       |
|1.0  |[1.0,1.0,1.0,34.0,0.0,126839.16666666667,7687.0]                             |0.0       |
|1.0  |[1.0,1.0,1.0,34.0,1.3333333333333333,103421.66666666667,3367.1666666666665]  |1.0       |
|1.0  |[1.0,1.0,1.0,34.0,1.6666666666666667,115795.66666666667,5100.0]              |1.0       |
|1.0  |[1.0,1.0,1.0,35.0,-1.6666666666666667,575.6666666666666,0.0]                 |0.0       |

Random Forest accuracy :  0.807145257028
'''

#Decision Trees model
dt = DecisionTreeClassifier(labelCol="label", featuresCol="features")
dt_model = dt.fit(training_data)
#Predict on the test data
predictions = dt_model.transform(test_data)
result = predictions.select("label", "features", "prediction")
result.where('label = 1.0').show(20, False)
result.where('label = 0.0').show(20, False)
print 'Dicision Tree accuracy : ',evaluator.evaluate(predictions)

'''
Dicision Tree accuracy :  0.805577332288
'''

#Naive Bayes model
nb = NaiveBayes(labelCol="label", featuresCol="features")
nb_model = nb.fit(training_data)
#Predict on the test data
predictions = nb_model.transform(test_data)
result = predictions.select("label", "features", "prediction")
result.where('label = 1.0').show(20, False)
result.where('label = 0.0').show(20, False)
print 'Naive Bayes accuracy : ',evaluator.evaluate(predictions)

'''
Naive Bayes accuracy :  0.545212470875
'''
