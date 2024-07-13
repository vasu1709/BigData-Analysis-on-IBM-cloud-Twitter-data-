#Loading data into a cloud object storage
import os, types
import pandas as pd
from botocore.client import Config
import ibm_boto3

def __iter__(self): return 0

# @hidden_cell
# The following code accesses a file in your IBM Cloud Object Storage. It includes your credentials.
# You might want to remove those credentials before you share the notebook.
cos_client = ibm_boto3.client(service_name='s3',
    ibm_api_key_id='v6lNGE0FJsWMr1ShsgLHNrKW2E1ixk7EKbniKuw9p3MV',
    ibm_auth_endpoint="https://iam.cloud.ibm.com/oidc/token",
    config=Config(signature_version='oauth'),
    endpoint_url='https://s3.private.eu-de.cloud-object-storage.appdomain.cloud')

bucket = 'bigdataanalysis-donotdelete-pr-txflnszhllxyij'
object_key = 'tweets.csv'

body = cos_client.get_object(Bucket=bucket,Key=object_key)['Body']
# add missing __iter__ method, so pandas accepts body as file-like object
if not hasattr(body, "__iter__"): body.__iter__ = types.MethodType( __iter__, body )

df_data_1 = pd.read_csv(body)
df_data_1.head()

#Converting pandas dataframe into pyspark dataframe
import pyspark
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import HashingTF, Tokenizer, StopWordsRemover
from pyspark.sql import SparkSession
#create Spark session
appName = "Sentiment Analysis in Spark"
spark = SparkSession \
    .builder \
    .appName(appName) \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()
df=spark.createDataFrame(df_data_1)
display(df)

#Data preprocessing

#handling duplicates
n1 = df.count()
print("number of original data rows: ", n1)
n2 = df.dropDuplicates().count()
print("number of data rows after deleting duplicated data: ", n2)
n3 = n1 - n2
print("number of duplicated data: ", n3)

#handling missing data
#Delete row if there is at least one (column) missing data
NoMissingValue = df.dropDuplicates().dropna(
    how="any", subset=["ItemID", "Sentiment"])
no_of_miss_val= n1 - NoMissingValue.count()
print("number of missing value rows: ", no_of_miss_val)

#take mean value
mean1 = df.groupBy().avg("ItemId").take(1)[0][0]
print("mean ItemID: ", mean1)
mean2 = df.groupBy().avg("Sentiment").take(1)[0][0]
print("mean Sentiment: ", mean2)

#drop duplicated data and fill missing data with mean value
TweetCleanData=df.fillna(
    {'ItemId': mean1, 'Sentiment': mean2})
df.groupBy().avg("ItemId").show()
df.describe('ItemId','Sentiment').show()
correlation = df.corr('ItemId', 'Sentiment')
print("correlation between itemid & sentiment ", correlation)

#Splitting data into training & testing
dividedData = df.randomSplit([0.7, 0.3]) 
train_set = dividedData[0] #index 0 = data training
test_set = dividedData[1] #index 1 = data testing
train_rows = train_set.count()
test_rows = test_set.count()
print ("Training data rows:", train_rows, "; Testing data rows:", test_rows)

#Prepare training data
tokenizer = Tokenizer(inputCol="SentimentText", outputCol="SentimentWords")
tokenizedTrain = tokenizer.transform(train_set)
tokenizedTrain.show(truncate=False, n=5)

#Removing Unnecessary words
swr = StopWordsRemover(inputCol=tokenizer.getOutputCol(), outputCol="MeaningfulWords")
SwRemovedTrain = swr.transform(tokenizedTrain)
SwRemovedTrain.show(truncate=False, n=5)

#Converting words feature into numerical feature
hashTF = HashingTF(inputCol=swr.getOutputCol(), outputCol="features")
numericTrainData = hashTF.transform(SwRemovedTrain).select('MeaningfulWords', 'features','Sentiment')
numericTrainData.show(truncate=False, n=3)

#Train our classifier model using training data
lr = LogisticRegression(labelCol="Sentiment", featuresCol="features", maxIter=10, regParam=0.01)
model = lr.fit(numericTrainData)
print ("Training is done!")

#Prepare testing data
tokenizedTest = tokenizer.transform(test_set)
SwRemovedTest = swr.transform(tokenizedTest)
numericTest = hashTF.transform(SwRemovedTest).select('Sentiment','MeaningfulWords', 'features')
numericTest.show(truncate=False, n=3)

#Predict testing data and calculate the accuracy model
prediction = model.transform(numericTest)
predictionFinal = prediction.select("Sentiment","MeaningfulWords", "prediction")
predictionFinal.show(n=4, truncate = False)
correctPrediction = predictionFinal.filter(predictionFinal['prediction'] == predictionFinal['Sentiment']).count()
totalData = predictionFinal.count()
print("correct prediction:", correctPrediction, ", total data:", totalData, ", accuracy:", correctPrediction/totalData)

#Text Preprocessing
import nltk
from nltk.corpus import stopwords
from  nltk.stem import SnowballStemmer
import re
from pyspark.sql import functions as func
nltk.download('stopwords')
stop_words = stopwords.words("english")
stemmer = SnowballStemmer("english")
text_cleaning_re = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"

def preprocess(text, stem=False):
    # Remove link,user and special characters
    text = re.sub(text_cleaning_re, ' ', str(text).lower()).strip()
    tokens = []
    for token in text.split():
        if token not in stop_words:
            if stem:
                tokens.append(stemmer.stem(token))
            else:
                tokens.append(token)
    return " ".join(tokens)

%%time
clean_text = func.udf(lambda x: preprocess(x), StringType())
df = df.withColumn('text_cleaned',clean_text(func.col("SentimentText")))

df.show(n=5)

#Data Visualization
#Word cloud
from wordcloud import WordCloud
import matplotlib.pyplot as plt
pandas_df = df.toPandas()
pandas_df.head()

#Positive Sentiment cloud
plt.figure(figsize = (20,16)) 
wc = WordCloud(max_words = 2000 , width = 1600 , height = 900).generate(" ".join(pandas_df[pandas_df["Sentiment"]==1.0].text_cleaned))
plt.imshow(wc , interpolation = 'bilinear')

#Negative Sentiment Cloud
plt.figure(figsize = (20,16)) 
wc = WordCloud(max_words = 2000 , width = 1600 , height = 900).generate(" ".join(pandas_df[pandas_df["Sentiment"]==0.0].text_cleaned))
plt.imshow(wc , interpolation = 'bilinear')

#Pie chart
positive_count = df.filter(df['Sentiment'] == 1).count()
negative_count = df.filter(df['Sentiment'] == 0).count()
labels = ['Positive', 'Negative']
sizes = [positive_count, negative_count]
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()
