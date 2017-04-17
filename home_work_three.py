from __future__ import print_function
import sys

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: home_work_three <google csv file name> <amazon csv file name>", file=sys.stderr)
        exit(-1)

    from pyspark import SparkContext, SparkConf

    conf = SparkConf().setAppName("HomeWorkThree")
    sc = SparkContext(conf=conf)

    googleFileName = sys.argv[1]
    amazonFileName = sys.argv[2]

    #Part 1
    googleData = sc.textFile(googleFileName).map(lambda x: x.replace('"',''))

    amazonData = sc.textFile(amazonFileName).map(lambda x: x.replace('"',''))

    print('Number of data points in Google dataset {0}'.format(googleData.count()))
    print('Number of data points in Amazon dataset {0}'.format(amazonData.count()))

    #Part 2
    import re
    REGEX_PATTERN = '^(.+),(.+),(.*),(.*),(.*)'

    # (a) Implement a function that takes a string and returns non-empty tokens by splitting using regular expressions.
    def getNonEmptyTokens(line):
        match = re.search(REGEX_PATTERN, line)
        if match and match.group(1) != 'id':
            product = '%s %s %s' % (match.group(2), match.group(3), match.group(4))
            return (match.group(1), product)

    parsedGoogleData = googleData.map(getNonEmptyTokens).filter(lambda x: x)
    parsedAmazonData = amazonData.map(getNonEmptyTokens).filter(lambda x: x)

    print('Number of data points in Google dataset after parsing {0}'.format(parsedGoogleData.count()))
    print('Number of data points in Amazon dataset after parsing {0}'.format(parsedAmazonData.count()))

    #(b) Stopwords are common (English) words that do not contribute much to the content or meaning of a document (e.g., "the", "a", "is", "to", etc.). Stopwords add noise to bag-of-words comparisons, so they are usually excluded. Using the included file "stopwords.txt", implement a function, an improved tokenizer that does not emit stopwords.
    #(c) Tokenize both Amazon and Google datasets. Here for each instance, tokenize the values corresponding to keys.

    stopwords = set(sc.textFile('/Users/balajirajaram/Desktop/IU/spring-2017/Spark_2_practice/data/stopwords.txt').collect())

    split_regex = r'\W+'

    def removeStopWordsAndToken(string):
            return [word for word in re.split(split_regex,string.lower()) if word and word not in stopwords]

    tokenizedGoogleData = parsedGoogleData.map(lambda x:(x[0],removeStopWordsAndToken(x[1])))
    tokenizedAmazonData = parsedAmazonData.map(lambda x:(x[0],removeStopWordsAndToken(x[1])))


    # Part 3
    # Write a function that takes a list of tokens and returns a dictionary mapping tokens to term frequency.
    # Note: You can use MLLIB for TF and IDF functions
    from collections import Counter

    #reference: http://www.tfidf.com
    def tf(tokens):
        return Counter(tokens)

    tokenizedGoogleData.map(lambda x: tf(x[1])).take(5)

    #Part 4
    #Combine the datasets to create a corpus. Each element of the corpus is a <key, value> pair where key is ID and value is associated tokens from two datasets combined.
    from operator import add
    corpus = tokenizedGoogleData.union(tokenizedAmazonData).reduceByKey(add)

    # Part 5
    # Write an IDF function that return a pair RDD where key is each unique token and value is corresponding IDF value. Plot a histogram of IDF values.
    #reference: http://www.tfidf.com

    def idfs(corpus):
        uniqueTokens = corpus.flatMap(lambda x:list(set(x[1])))
        uniqueTokenCounts = sc.parallelize(uniqueTokens.countByValue().items())
        N = corpus.map(lambda x:x[0]).distinct().count()
        tokenIDFPairRDD = uniqueTokenCounts.map(lambda x:(x[0],float(N*(x[1])**(-1))))
        return tokenIDFPairRDD

    corpusIdf = idfs(corpus)

    import matplotlib.pyplot as plt

    corpusIdfValues = corpusIdf.map(lambda s: s[1]).collect()
    fig = plt.figure(figsize=(8,3))
    plt.hist(corpusIdfValues, 50, log=True)
    plt.savefig('IDF_Histogram.png')

    # Part 6
    # Write a function which does the following on entire corpus:
    # (a) Calculate token frequencies for tokens
    # (b) Create a Python dictionary where each token maps to the token's frequency times the token's IDF weight

    def tfidf(tokens, idfs):
        tfs = tf(tokens)
        tfIdfDict={}
        for token in tfs.keys():
          tfIdfDict[token]=tfs[token]*idfs[token]
        return tfIdfDict

    corpusIdfMap = corpusIdf.collectAsMap()
    finalTFIDF = corpus.map(lambda x: tfidf(x[1],corpusIdfMap))

    print(finalTFIDF.take(5))
