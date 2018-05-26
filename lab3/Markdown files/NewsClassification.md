
# SHUBHAM SHAILESH PANDEY, UBID - spandey4

### The method below reads the text files from the subfolders inside the Data folder and makes a Data Frame out of it. The Dataframe contains 3 columns after this - 
### 'value' - This column contains an entire article
### 'filename' - This column contains the filepath of the article 
### 'Category'- The category to which the article belongs to (Business,Politics,Sports or Technology)
### 'label' - This column maps a unique integer based on the article's category



```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import input_file_name,col,lit,split,lower
from pyspark.ml.feature import StopWordsRemover
from pyspark.ml.feature import HashingTF, IDF
from pyspark.sql import functions as F
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import regexp_extract

def make_df():
    df1 = spark.read.text("./Data/Technology")
    df1 = df1.withColumn("filename", input_file_name())
    df1 = df1.withColumn("Category",lit("Technology"))
    df1 = df1.withColumn("label",lit(0))

    df2 = spark.read.text("./Data/Sports")
    df2 = df2.withColumn("filename", input_file_name())  
    df2 = df2.withColumn("Category",lit("Sports"))
    df2 = df2.withColumn("label",lit(1))

    df3 = spark.read.text("./Data/Business")
    df3 = df3.withColumn("filename", input_file_name())  
    df3 = df3.withColumn("Category",lit("Business"))
    df3 = df3.withColumn("label",lit(2))

    df4 = spark.read.text("./Data/Politics")
    df4 = df4.withColumn("filename", input_file_name())
    df4 = df4.withColumn("Category",lit("Politics"))
    df4 = df4.withColumn("label",lit(3))

    df = df1.union(df2)
    df = df.union(df3)
    df = df.union(df4)
    return df
```

### The method below lists some regex rules needed to clean the article words


```python
def regex_rules(df):
    df = df.withColumn('words', F.regexp_replace('value', 'http+s*\:+\/\/[a-zA-Z\.\/0-9]+ *', ''))
    df = df.withColumn('words', F.regexp_replace('words', ' {1}[a-zA-Z0-9]+…$', ''))
    df = df.withColumn('words', F.regexp_replace('words', '[\)\( \/\.-][0-9]+[ \)\/\.\(-]', ' '))
    df = df.withColumn('words', lower(col('words')))
    df = df.withColumn('words', F.regexp_replace('words', "'s", ''))
    df = df.withColumn('words', F.regexp_replace('words', '[^a-zA-Z0-9 ]+', ' '))
    df = df.withColumn('words', F.regexp_replace('words', '[ ][0-9]+ ', ''))
    df = df.withColumn('words', split('words', "\s+"))
    return df
```

### The method below removes the stop words mentioned in the array. They are not essential to the data we need to classify. I used the StopWordsRemover library to do this.


```python
def stopwords_remove(df):
    stopwords = ['a','an','the','and','is','are','was','were','what','them','had','some','ca',
             'why','when','where','who','whose','which','that','off','ever','many','ve',
             'those','this','those','but','so','thus','again','therefore','its','both',
             'like','in','on','up','down','under','over','i','we','they','while','okay',
             'he','them','their','there','us','of','you','your','us','our','mine','mr',
             'such','am','to','too','for','from','since','until','between','she','own',
             'my','not','if','as', 'well','youre','hadnt','havent','wont','q','se','ok',
             'very','have','it','be','been','has','having','his', 'her','never','above',
             'should','would', 'could','just', 'about','do','doing','does','did','la','ha',
             'go','going','goes','being','with', 'yes', 'no','how','before','than','d',
             'after','any','here','out','now','then','got','into','all','cant','or','ya',
             'despite','beyond','further','wanna', 'want','gonna','isnt', 'at','also','lo',
             'because','due','heres','try','said','says','will','shall','link','asked',
             'more','less','often','lol','maybe','perhaps','quite','even','him','by','n',
             'among','can','may','most','took','during','me','told','might','hi','es','l',
             'theyll','use','u','whats','couldnt','wouldnt','see','im','dont','x','de',
             'doesnt','shouldnt', 'hes','thats','let','lets','get','gets','en','co','k',
             'whats','s','say','via','youll','wed','theyd','youd','w','m','hey','hello',
             'youve','theyve','weve','theyd','youd','ive','were','ill','yet','b','rt',
             'id','o','r','z','um','em','seen','didnt','r','e','t','c','y','only','v',
             'arent','werent','hasnt','mostly','much','ago','wasnt','aint','nope','p',
             'll','ja','al','el','gt','cs','si','didnt','re','f','fo','j','ni','tr','il',
             'rt','http','https','amp']
    remover = StopWordsRemover(inputCol="words", outputCol="filtered_words", stopWords = stopwords)
    df = remover.transform(df)
    return df
```

### The drop_cols method below drop unnecessary columns from the data frame generated.
### The tf_idf method generates a column of features based on the word and document frequencies.
### These features will be used for training the data using the various classification methods.
### After lot of parameter tuning, I observed the best results when I use 13100 features 


```python
def drop_cols(df):
    df = df.withColumn('file', regexp_extract('filename', '^.*/(.*)', 1)) \
        .drop('filename')
    df = df.drop('rawFeatures')
    return df

def tf_idf(df):
    hashingTF = HashingTF(inputCol="filtered_words", outputCol="rawFeatures", numFeatures=13100)
    featurizedData = hashingTF.transform(df)
    idf = IDF(inputCol="rawFeatures", outputCol="features")
    idfModel = idf.fit(featurizedData)
    df = idfModel.transform(featurizedData)
    return df
```

### The method below classifies data using Random Forest Classifier. I show 30 predictions to show how much they conformed to the actual labels.


```python
def random_forest(trainingData,testData):
    from pyspark.ml.classification import RandomForestClassifier
    print("Random Forest Classifier")
    rf = RandomForestClassifier(labelCol="label", featuresCol="features")
    model = rf.fit(trainingData)
    predictions = model.transform(testData)
    # Select example rows to display.
    predictions.select("value","Category","probability","label","prediction") \
        .orderBy("probability", ascending=False) \
        .show(n = 30, truncate = 30)
    evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
    print("Accuracy :- " + str(100*evaluator.evaluate(predictions)) +" %"+"\n")
    
    
```

### The method below classifies data using Naive Bayes Classifier. I show 30 predictions to show how much they conformed to the actual labels.


```python
def naive_bayes(trainingData,testData):
    from pyspark.ml.classification import NaiveBayes
    print("Naive Bayes Classifier")
    nb = NaiveBayes(smoothing=1)
    model = nb.fit(trainingData)
    predictions = model.transform(testData)
    predictions.select("value","Category","probability","label","prediction") \
        .orderBy("probability", ascending=False) \
        .show(n = 30, truncate = 30)
    evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
    print("Accuracy :- " + str(100*evaluator.evaluate(predictions)) +" %"+"\n")
    

```

### The method classifies data using Logistic Regression. I show 30 predictions to show how much they conformed to the actual labels.


```python
def logistic_regression(trainingData,testData):
    from pyspark.ml.classification import LogisticRegression
    print("Logistic Regression")
    lr = LogisticRegression(maxIter=10, regParam=0.32, elasticNetParam=0)
    lrModel = lr.fit(trainingData)
    predictions = lrModel.transform(testData)
    predictions.select("value","Category","probability","label","prediction") \
            .orderBy("probability", ascending=False) \
            .show(n = 30, truncate = 30)
    evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
    print("Accuracy :- " + str(100*evaluator.evaluate(predictions)) +" %"+"\n")
```

### The script below sets up a spark session, makes a dataframe from the articles read, applies regex rules to the dataframe to clean the articles, removes stop words, extracts features using TF-IDF algorithm, drops unnecessary columns, splits the data into training and test data in [80:20] ratio, fits the training data using Naive Bayes and Logistic Regression models and finally predicts the labels for test data.


```python
spark = SparkSession \
        .builder \
        .getOrCreate()
    
df = make_df()
df = regex_rules(df)
df = stopwords_remove(df)
df = tf_idf(df)
df = drop_cols(df)
df.show(5)

(trainingData, testData) = df.randomSplit([0.8, 0.2], seed = 100000)

print("Training Data Count: " + str(trainingData.count()))
print("Test Data Count: " + str(testData.count())+"\n")

#random_forest(trainingData,testData)
naive_bayes(trainingData,testData)
logistic_regression(trainingData,testData)
```

    +--------------------+----------+-----+--------------------+--------------------+--------------------+----------------+
    |               value|  Category|label|               words|      filtered_words|            features|            file|
    +--------------------+----------+-----+--------------------+--------------------+--------------------+----------------+
    |SAN FRANCISCO — T...|Technology|    0|[san, francisco, ...|[san, francisco, ...|(13100,[1,13,18,2...|Technology52.txt|
    |It may not qualif...|Technology|    0|[it, may, not, qu...|[qualify, lightni...|(13100,[1,3,8,9,1...|Technology22.txt|
    |Robert O. Work, t...|Technology|    0|[robert, o, work,...|[robert, work, ve...|(13100,[12,48,108...| Technology2.txt|
    |Andrew S. Grove, ...|Technology|    0|[andrew, s, grove...|[andrew, grove, l...|(13100,[8,13,36,5...| Technology7.txt|
    |SEATTLE — When Gl...|Technology|    0|[seattle, when, g...|[seattle, glenn, ...|(13100,[13,38,80,...|Technology24.txt|
    +--------------------+----------+-----+--------------------+--------------------+--------------------+----------------+
    only showing top 5 rows
    
    Training Data Count: 194
    Test Data Count: 43
    
    Naive Bayes Classifier
    +------------------------------+----------+------------------------------+-----+----------+
    |                         value|  Category|                   probability|label|prediction|
    +------------------------------+----------+------------------------------+-----+----------+
    |SAN FRANCISCO — Mark Zucker...|Technology|[1.0,0.0,4.79913831054448E-...|    0|       0.0|
    |SEATTLE — When Glenn Kelman...|Technology|[1.0,0.0,5.289680294393272E...|    0|       0.0|
    |ORION TOWNSHIP, Mich. — Ten...|Technology|[1.0,0.0,9.362268341373143E...|    0|       0.0|
    |Google’s Pixel, the first s...|Technology|[1.0,0.0,1.5908291879336067...|    0|       0.0|
    |HONG KONG — When the United...|Technology|[1.0,0.0,7.445832538668339E...|    0|       0.0|
    |It was the biggest known br...|Technology|[1.0,0.0,5.887054833715122E...|    0|       0.0|
    |SAN FRANCISCO — The resound...|Technology|             [1.0,0.0,0.0,0.0]|    0|       0.0|
    |Jeffry Melnick didn’t want ...|  Business|[0.9999679180929917,3.37938...|    2|       0.0|
    |LOS ANGELES — Charles H. Ri...|  Business|[1.2607450008136603E-36,1.2...|    2|       2.0|
    |WASHINGTON — A federal law ...|    Sports|[1.0227266800038032E-48,1.0...|    1|       1.0|
    |On Aug. 25, 2015, a Swiss p...|    Sports|[7.444753071453243E-50,1.0,...|    1|       1.0|
    |(Reuters) - Pub operator JD...|  Business|[1.4467949646774072E-69,2.2...|    2|       2.0|
    |(Reuters) - MetLife Inc rep...|  Business|[2.3161524139836607E-75,2.1...|    2|       2.0|
    |GANGNEUNG, South Korea — It...|    Sports|[1.8639739023208585E-104,1....|    1|       1.0|
    |LOS ANGELES — The global so...|  Business|[6.272082463549182E-107,4.3...|    2|       2.0|
    |LOUISVILLE, Ky. — Mick Ruis...|    Sports|[1.373897568832535E-115,1.0...|    1|       1.0|
    |WASHINGTON — Bobby Amos sto...|  Politics|[3.2275534659418613E-128,0....|    3|       3.0|
    |PHILADELPHIA — When T. J. M...|    Sports|[8.321537797321434E-144,1.0...|    1|       1.0|
    |OAKLAND, Calif. — Stephen C...|    Sports|[5.769253986331905E-150,1.0...|    1|       1.0|
    |A group of former pro footb...|    Sports|[9.515983836223461E-152,1.0...|    1|       1.0|
    +------------------------------+----------+------------------------------+-----+----------+
    only showing top 20 rows
    
    Accuracy :- 97.69073847409221 %
    
    Logistic Regression
    +------------------------------+----------+------------------------------+-----+----------+
    |                         value|  Category|                   probability|label|prediction|
    +------------------------------+----------+------------------------------+-----+----------+
    |SAN FRANCISCO — The resound...|Technology|[0.9226459740373305,0.02436...|    0|       0.0|
    |HONG KONG — When the United...|Technology|[0.7835012672852633,0.03156...|    0|       0.0|
    |It was the biggest known br...|Technology|[0.7591213111800397,0.07235...|    0|       0.0|
    |Google’s Pixel, the first s...|Technology|[0.7461168366857824,0.18681...|    0|       0.0|
    |ORION TOWNSHIP, Mich. — Ten...|Technology|[0.7404059616492071,0.05477...|    0|       0.0|
    |Jeffry Melnick didn’t want ...|  Business|[0.549857104547079,0.140986...|    2|       0.0|
    |SAN FRANCISCO — Mark Zucker...|Technology|[0.4768721818095614,0.03598...|    0|       0.0|
    |MOON, Pa. — When the Pittsb...|    Sports|[0.2732973690639352,0.69114...|    1|       1.0|
    |GANGNEUNG, South Korea — It...|    Sports|[0.25910286919401776,0.2918...|    1|       3.0|
    |On Aug. 25, 2015, a Swiss p...|    Sports|[0.25673249689224836,0.6606...|    1|       1.0|
    |WASHINGTON — Bobby Amos sto...|  Politics|[0.23625991340227837,0.0495...|    3|       3.0|
    |(Reuters) - MetLife Inc rep...|  Business|[0.23356457610172324,0.1046...|    2|       2.0|
    |WASHINGTON — Judge Neil M. ...|  Politics|[0.23261941403534853,0.1062...|    3|       3.0|
    |PHILADELPHIA — When T. J. M...|    Sports|[0.2082064493238247,0.68006...|    1|       1.0|
    |(Reuters) - Pub operator JD...|  Business|[0.18882563997812563,0.1093...|    2|       2.0|
    |WASHINGTON — When it comes ...|  Politics|[0.1880020312338311,0.13672...|    3|       3.0|
    |A group of former pro footb...|    Sports|[0.17919455392806535,0.4229...|    1|       1.0|
    |LOS ANGELES — Charles H. Ri...|  Business|[0.17853674564003633,0.2739...|    2|       2.0|
    |WASHINGTON — A federal law ...|    Sports|[0.17462901963632846,0.3733...|    1|       1.0|
    |SEATTLE — When Glenn Kelman...|Technology|[0.1636480465056957,0.02280...|    0|       2.0|
    +------------------------------+----------+------------------------------+-----+----------+
    only showing top 20 rows
    
    Accuracy :- 93.0198105081826 %
    
    

### The same method for making a dataframe is repeated for the unknown data set read from Unknown folder


```python
def make_df_unknown():
    df1 = spark.read.text("./Unknown/Technology")
    df1 = df1.withColumn("filename", input_file_name())
    df1 = df1.withColumn("Category",lit("Technology"))
    df1 = df1.withColumn("label",lit(0))

    df2 = spark.read.text("./Unknown/Sports")
    df2 = df2.withColumn("filename", input_file_name())  
    df2 = df2.withColumn("Category",lit("Sports"))
    df2 = df2.withColumn("label",lit(1))

    df3 = spark.read.text("./Unknown/Business")
    df3 = df3.withColumn("filename", input_file_name())  
    df3 = df3.withColumn("Category",lit("Business"))
    df3 = df3.withColumn("label",lit(2))

    df4 = spark.read.text("./Unknown/Politics")
    df4 = df4.withColumn("filename", input_file_name())
    df4 = df4.withColumn("Category",lit("Politics"))
    df4 = df4.withColumn("label",lit(3))

    df = df1.union(df2)
    df = df.union(df3)
    df = df.union(df4)
    return df
```

### The same script for making predictions is repeated for the unknown dataframe.


```python
unknownData = make_df_unknown()
unknownData = regex_rules(unknownData)
unknownData = stopwords_remove(unknownData)
unknownData = tf_idf(unknownData)
unknownData = drop_cols(unknownData)
unknownData.show(5)
print(unknownData.count())
naive_bayes(trainingData,unknownData)
logistic_regression(trainingData,unknownData)
spark.stop()
```

    +--------------------+----------+-----+--------------------+--------------------+--------------------+----------------+
    |               value|  Category|label|               words|      filtered_words|            features|            file|
    +--------------------+----------+-----+--------------------+--------------------+--------------------+----------------+
    |The scene opened ...|Technology|    0|[the, scene, open...|[scene, opened, r...|(13100,[29,35,92,...|Technology61.txt|
    |WASHINGTON — Thre...|Technology|    0|[washington, thre...|[washington, thre...|(13100,[35,94,108...|Technology80.txt|
    |For the past seve...|Technology|    0|[for, the, past, ...|[past, several, y...|(13100,[33,35,77,...|Technology78.txt|
    |Here’s a question...|Technology|    0|[here, s, a, ques...|[question, hoping...|(13100,[35,95,151...|Technology77.txt|
    |Each Friday, Farh...|Technology|    0|[each, friday, fa...|[each, friday, fa...|(13100,[8,13,35,9...|Technology62.txt|
    +--------------------+----------+-----+--------------------+--------------------+--------------------+----------------+
    only showing top 5 rows
    
    85
    Naive Bayes Classifier
    +------------------------------+----------+------------------------------+-----+----------+
    |                         value|  Category|                   probability|label|prediction|
    +------------------------------+----------+------------------------------+-----+----------+
    |Galleries like American Med...|Technology|[1.0,8.549294886753447E-61,...|    0|       0.0|
    |When Devin Patrick Kelley t...|  Business|[1.0,9.249324976179852E-113...|    2|       0.0|
    |Q. My computer’s hard drive...|Technology|[1.0,9.603285844207647E-117...|    0|       0.0|
    |CULVER CITY, Calif. — It wa...|  Business|[1.0,1.4619197275950615E-12...|    2|       0.0|
    |Airbnb has capitulated to t...|Technology|[1.0,2.8763693279135873E-17...|    0|       0.0|
    |HONG KONG — As one of China...|Technology|[1.0,7.822699261895335E-179...|    0|       0.0|
    |SAN FRANCISCO — For weeks, ...|Technology|[1.0,3.821523380888397E-203...|    0|       0.0|
    |SEATTLE — The other week, 2...|Technology|[1.0,4.205423781264313E-223...|    0|       0.0|
    |SAN FRANCISCO — Attorney Ge...|Technology|[1.0,2.924655649381282E-241...|    0|       0.0|
    |Here’s a question I’m hopin...|Technology|[1.0,2.2937351064018492E-25...|    0|       0.0|
    |LUGANO, Switzerland — Jürge...|Technology|[1.0,6.606005106444626E-268...|    0|       0.0|
    |Each Saturday, Farhad Manjo...|Technology|[1.0,8.71316464030493E-275,...|    0|       0.0|
    |For 18 days last month, a t...|Technology|[1.0,1.58E-322,9.5750613444...|    0|       0.0|
    |Each Friday, Farhad Manjoo ...|Technology|[1.0,0.0,2.2740951834269626...|    0|       0.0|
    |SAN FRANCISCO — The phone c...|Technology|[1.0,0.0,1.879795139034764E...|    0|       0.0|
    |SAN FRANCISCO — Travis Kala...|Technology|[1.0,0.0,1.5584974737844475...|    0|       0.0|
    |SAN FRANCISCO — The interne...|Technology|[1.0,0.0,1.2497985607369106...|    0|       0.0|
    |WASHINGTON — Twitter said o...|Technology|[1.0,0.0,2.351661481496929E...|    0|       0.0|
    |WASHINGTON — Three years ag...|Technology|[1.0,0.0,7.616339495089694E...|    0|       0.0|
    |MOUNTAIN VIEW, Calif. — If ...|Technology|[1.0,0.0,1.137356217026606E...|    0|       0.0|
    +------------------------------+----------+------------------------------+-----+----------+
    only showing top 20 rows
    
    Accuracy :- 92.8220234642897 %
    
    Logistic Regression
    +------------------------------+----------+------------------------------+-----+----------+
    |                         value|  Category|                   probability|label|prediction|
    +------------------------------+----------+------------------------------+-----+----------+
    |The scene opened on a room ...|Technology|[0.9657986393252186,0.00681...|    0|       0.0|
    |For the past several years,...|Technology|[0.9615257637450412,0.00552...|    0|       0.0|
    |WASHINGTON — Twitter said o...|Technology|[0.866805340386594,0.027321...|    0|       0.0|
    |SAN FRANCISCO — The interne...|Technology|[0.7532436655210883,0.07794...|    0|       0.0|
    |MOUNTAIN VIEW, Calif. — If ...|Technology|[0.7507133777717101,0.12234...|    0|       0.0|
    |WASHINGTON — Three years ag...|Technology|[0.7427838701632212,0.06230...|    0|       0.0|
    |SAN FRANCISCO — For weeks, ...|Technology|[0.6977259799222686,0.08551...|    0|       0.0|
    |Each Saturday, Farhad Manjo...|Technology|[0.6514371295122293,0.16679...|    0|       0.0|
    |SAN FRANCISCO — Attorney Ge...|Technology|[0.6500775097745726,0.10192...|    0|       0.0|
    |For 18 days last month, a t...|Technology|[0.6047126460211742,0.13404...|    0|       0.0|
    |SAN FRANCISCO — The phone c...|Technology|[0.5976867492415747,0.08056...|    0|       0.0|
    |Q. My computer’s hard drive...|Technology|[0.5834051746163847,0.15897...|    0|       0.0|
    |Each Friday, Farhad Manjoo ...|Technology|[0.5653032798278417,0.08307...|    0|       0.0|
    |CULVER CITY, Calif. — It wa...|  Business|[0.5603985993749571,0.24529...|    2|       0.0|
    |SEATTLE — The other week, 2...|Technology|[0.5563335856810702,0.09115...|    0|       0.0|
    |LUGANO, Switzerland — Jürge...|Technology|[0.5438702631346882,0.15612...|    0|       0.0|
    |Galleries like American Med...|Technology|[0.5252966400009879,0.19040...|    0|       0.0|
    |Airbnb has capitulated to t...|Technology|[0.5232993292547028,0.15052...|    0|       0.0|
    |Here’s a question I’m hopin...|Technology|[0.49625985368099423,0.1324...|    0|       0.0|
    |HONG KONG — As one of China...|Technology|[0.4730507346976421,0.13668...|    0|       0.0|
    +------------------------------+----------+------------------------------+-----+----------+
    only showing top 20 rows
    
    Accuracy :- 87.07626640399751 %
    
    

### References used for the project - 
### https://spark.apache.org/docs/2.2.0/ml-features.html
### https://spark.apache.org/docs/2.1.0/ml-classification-regression.html#naive-bayes
### https://spark.apache.org/docs/2.1.0/ml-classification-regression.html#random-forests
### https://spark.apache.org/docs/2.1.0/ml-classification-regression.html#logistic-regression
