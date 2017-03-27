
# coding: utf-8

# In[7]:

import pyspark
from pyspark import SparkContext
from pyspark.mllib.fpm import FPGrowth
from operator import itemgetter
from pyspark.sql import SparkSession

SparkContext.setSystemProperty('spark.executor.memory','6g')
sc = pyspark.SparkContext('local[*]')

spark = SparkSession     .builder     .appName("Python Spark SQL basic example")     .config("spark.some.config.option", "some-value")     .getOrCreate()


# In[8]:

import re
from pyspark.ml.feature import Tokenizer, RegexTokenizer
from pyspark.sql.functions import col, udf
from pyspark.sql.types import IntegerType

rdd = sc.textFile('publications.txt')
small_rdd = rdd.sample(False, 1e-3)

# NOTE: Replace small_rdd with rdd to run on the full sample
title_rdd = small_rdd.filter(
            lambda l: re.match('^#\*(.*)',l)).map(
            lambda l: re.match('^#\*(.*)',l).group(1)).filter(lambda l: l.strip() != '')
title_list = title_rdd.map(lambda l: (0,l.lower()))
# Using regex toxenizer to remove unwanted puntuations
regexTokenizer = RegexTokenizer(inputCol="sentence", outputCol="words", pattern="\\W")


# In[60]:

from pyspark.ml.feature import CountVectorizer
# Generic count vectorizer which produces output as 'features'
def count_vectorizer_generic(data_frame,vocab_size,input_col):
    print('Count Vectorizer Result with output column features')
    cv_generic = CountVectorizer(inputCol=input_col, outputCol="features", vocabSize=vocab_size)
    model_generic = cv_generic.fit(data_frame)
    result_generic = model_generic.transform(data_frame)
    result_generic.show()
    print('\n')
    return (result_generic,model_generic)


# In[66]:

from pyspark.ml.feature import PCA
from pyspark.ml.linalg import Vectors
# Generic pca which produces output as 'pcaFeatures' output column
def pca_generic(data, dimens, input_col, output_col = "pcaFeatures"):
    print('PCA Result with dimentions = '+str(dimens)+' with output column pcaFeatures')
    pca_generic = PCA(k=dimens,inputCol=input_col, outputCol=output_col)
    pca_model_generic = pca_generic.fit(data)
    result_pca_generic = pca_model_generic.transform(data)
    result_pca_generic.show()
    print('\n')
    return result_pca_generic,pca_model_generic


# In[62]:

from pyspark.ml.feature import HashingTF, IDF, Tokenizer
# Generic TF_IDF which produces 'idfFeatures' as output column
def tf_idf(result_frame,num_features):
    print('TF_IDF result with output as idfFeatures')
    idf_generic = IDF(inputCol="features", outputCol="idfFeatures")
    idfModel_generic = idf_generic.fit(result_frame)
    rescaled_data = idfModel_generic.transform(result_frame)
    rescaled_data.show(truncate=True)
    return rescaled_data


# In[63]:

import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from pyspark.ml.feature import PCA
from pyspark.ml.linalg import Vectors
# Generic Scatter Plotting Code
def scatter_plot(result_gen,color):
    r = result_gen.select('pcaFeatures').rdd.map(lambda row : row[0])
    x = list(r.map(lambda l: l[0]).collect())
    y = list(r.map(lambda l: l[1]).collect())
    plt.scatter(x, y,c=color,alpha=0.5,)
    plt.show()


# In[64]:

import matplotlib.pyplot as plt
import numpy as np
# explainedVariance gives eigenvalues for any pca model
def plot_eigenvalues(pca_model):
    eigenvalues = pca_model.explainedVariance
    plt.plot(eigenvalues)
    return eigenvalues
# Generic funtion to plot cumulative variances
def cumsum_variances(eigenvalues):
    cumsum_variances = np.cumsum(eigenvalues) / np.sum(eigenvalues)
    plt.plot(cumsum_variances)
    print(cumsum_variances)
    
    


# In[68]:

# a. Transform titles to word count vectors. Truncate your (sparse) vectors to the
# 1000 most frequent words and perform PCA with 50 components on the counts.
# from pyspark.ml.feature import CountVectorizer

# Input data: Each row is a bag of words with a ID.
dataframe_titles = regexTokenizer.transform(spark.createDataFrame(title_list, ["id", "sentence"]))
cv_titles_result,cv_titles_model = count_vectorizer_generic(dataframe_titles,1000,"words")
pca_titles_cv,pca_model_titles = pca_generic(cv_titles_result,50,"features")



# In[69]:

# b. Plot the eigenvalues of the principal components. Calculate how many components
# are needed to explain 50% of the total variance?
eigenvals = plot_eigenvalues(pca_model_titles);


# In[70]:

cumsum_variances(eigenvals)


# We need 7 componenets to explain 50% of total variance

# In[71]:

# c. Identify which words are important in each of the principal components. To do
# so, take the sum of squares of each of the component vectors to check how they
# are normalized. For each component, then print out the words for which the
# absolute value of the component is larger than 0.20 of the norm.

def most_imp_words(pca_model,cv_model):
    vocab = cv_model.vocabulary
    eigenvectors = pca_model.pc.toArray().T
    for k,ev in enumerate(eigenvectors):
        ws = [vocab[w] for (w,c) in enumerate(np.abs(ev)) if c > 0.2]
        print(k,ws)

    
most_imp_words(pca_model_titles,cv_titles_model)


# In[72]:

# d. Make a scatter plot of some reasonably sized sample (1k-10k titles). Explain
# the structure (or lack thereof) you see based on your results from item b-c.
pca_titles_cv_scatter,pca_model_titles_scatter = pca_generic(cv_titles_result,2,"features")
scatter_plot(pca_titles_cv_scatter,['blue'])


# ### Structure of Scatter Plot
# So scatter plot looks unstructured because of lot of anamolies in data like stopwords, which we will remove in the later state. This is also evident from part(c) since stopwords like 'of','the','for'...are really evident in many principal componenets and they really should not be carrying any weight. Here the variance capture of first two components is 21% which will result in more groups with less structured data since data captured is more spread out.

# In[73]:

# e. Run a preprocessing step to remove stop words (a list of stop words is provided
# which is identical to the list used in Spark). Rerun steps b-d and evaluate
# whether this has improved your representation.
from pyspark.ml.feature import StopWordsRemover

stopwords_rdd = sc.textFile('stopwords_english.txt').collect()
# Configure the stopword remover
# NOTE: column 'filtered' will always represent words after stopwords are removed
remover = StopWordsRemover(inputCol="words", outputCol="filtered",stopWords=stopwords_rdd)
filtered_dataframe = remover.transform(dataframe_titles)
cv_stopwords_result,cv_stopwords_model = count_vectorizer_generic(filtered_dataframe,1000,"filtered")
pca_stopwords_cv_result,pca_stopwords_titles_model = pca_generic(cv_stopwords_result,50,"features")



# In[74]:

# Plot the eigenvalues of the principal components. Calculate how many components
# are needed to explain 50% of the total variance?
eigenvals_stopwords = plot_eigenvalues(pca_stopwords_titles_model);


# In[75]:

cumsum_variances(eigenvals_stopwords)


# We would need 15 principal compenents here to explain 50% of total variance

# In[76]:

# Identify which words are important in each of the principal components. To do
# so, take the sum of squares of each of the component vectors to check how they
# are normalized. For each component, then print out the words for which the
# absolute value of the component is larger than 0.20 of the norm.

most_imp_words(pca_stopwords_titles_model,cv_stopwords_model)


# In[77]:

# Make a scatter plot of some reasonably sized sample (1k-10k titles). Explain
# the structure (or lack thereof) you see based on your results from item b-c.
pca_titles_stopwords_scatter,pca_model_stopwords_scatter = pca_generic(cv_stopwords_result,2,"features")
scatter_plot(pca_titles_stopwords_scatter,['red'])


# ### Structure of Scatter Plot
# Scatter plot now after removing stopwords looks structured and similar compnents show up in same group/community suggesting some pattern in values. From part(c) here we can also see that there are no stopwords in the principal compnenets hence only relevant words are given weight, results in scatter plot are more effective. Variance capture by first two components is approx 12% which is less compared to previous with stopwords hence data is less spread out and results in better structure with lesser groups.
# 

# In[78]:

# f. One of the issues with text is that the variance in how often a word appears is
# often correlated with the base frequency with which a word appears. To account
# for this, the word counts are often reweighted using the term frequency, inverse
# document frequency (TF-IDF) scheme:
tfidf_rescaled_result = tf_idf(cv_stopwords_result,1000)
pca_titles_tfidf_result,pca_titles_tfidf_model = pca_generic(tfidf_rescaled_result,50,"idfFeatures")


# In[79]:

# Plot the eigenvalues of the principal components. Calculate how many components
# are needed to explain 50% of the total variance?
eigenvals_tfidf = plot_eigenvalues(pca_titles_tfidf_model);


# In[81]:

cumsum_variances(eigenvals_tfidf)


# We need 20 principal componenets to describe 50% of the principal components

# In[82]:

# Identify which words are important in each of the principal components. To do
# so, take the sum of squares of each of the component vectors to check how they
# are normalized. For each component, then print out the words for which the
# absolute value of the component is larger than 0.20 of the norm.

most_imp_words(pca_titles_tfidf_model,cv_stopwords_model)


# In[83]:

# Make a scatter plot of some reasonably sized sample (1k-10k titles). Explain
# the structure (or lack thereof) you see based on your results from item b-c.

pca_titles_tfidf_scatter,pca_model_tfidf_scatter = pca_generic(tfidf_rescaled_result,2,"idfFeatures")
scatter_plot(pca_titles_tfidf_scatter,['blue'])


# ### Structure of Scatter Plot
# 
# Tf-idf scatter plot looks less structured than the count vector plot for the same principal components. Tf-idf is not able to seperate groups as effectively as count vector does. Results of the scatter plot look to almost form one community or group. Here variance captureby first two components is 7% and data points nearly show no groups since data points are close to each other
# 

# ### PART-2
# In the second part of this exercise, we will look at subsets of the data. To construct
# these subsets, first construct two lists of titles:
# • Titles for machine learning papers published at ‘NIPS’
# • Titles for database papers published at ‘VLDB’

# In[84]:

file = open('publications.txt')

NIPS_list = list()
VLDB_list = list()

title_keep = ''

for line in file:
        if line.startswith('#*'):
            title_keep = line[2:].strip().replace(".","")
                      
        elif line.startswith('#c'):
            publication_venue = line[2:].strip()
            if publication_venue == 'NIPS' and title_keep != '':
                NIPS_list.append(title_keep)
            elif publication_venue == 'VLDB' and title_keep != '':
                VLDB_list.append(title_keep)
                
            title_keep = ''
            


# In[85]:

title_list_1 = list(map(lambda l: (0,l.lower()), NIPS_list))
title_list_2 = list(map(lambda l: (1,l.lower()), VLDB_list))

title_list_1.extend(title_list_2)

dataframe_conferences = regexTokenizer.transform(spark.createDataFrame(title_list_1, ["id", "sentence"]))
filtered_dataframe_conferences = remover.transform(dataframe_conferences)
cv_conference_result,cv_conference_model = count_vectorizer_generic(filtered_dataframe_conferences,1000,"filtered")
pca_conference_cv,pca_conference_model = pca_generic(cv_conference_result,50,"features")



# In[86]:

# Plot the eigenvalues of the principal components. Calculate how many components
# are needed to explain 50% of the total variance?
eigenvals_conference_cv = plot_eigenvalues(pca_conference_model);


# In[87]:

cumsum_variances(eigenvals_conference_cv)


# We need 13 principal componenets to describe 50% of the principal components

# In[88]:

# Identify which words are important in each of the principal components. To do
# so, take the sum of squares of each of the component vectors to check how they
# are normalized. For each component, then print out the words for which the
# absolute value of the component is larger than 0.20 of the norm.

most_imp_words(pca_conference_model,cv_conference_model)


# In[102]:

# Now make a scatter plot of these two principal components, showing the titles from each subset in different colors.

pca_confernece_scatter,pca_model_conference_scatter = pca_generic(cv_conference_result,2,"features")
# Generic function to seperate features of two list and plot them
def list_plot(pca_result):
    pca_VLDB = pca_result.select("id", "pcaFeatures").rdd.filter(lambda r: r[0] == 1)
    pca_NIPS = pca_result.select("id", "pcaFeatures").rdd.filter(lambda r: r[0] == 0)
    x_NIPS = pca_NIPS.map(lambda l: l[1][0]).collect()
    y_NIPS = pca_NIPS.map(lambda l: l[1][1]).collect()
    x_VLDB = pca_VLDB.map(lambda l: l[1][0]).collect()
    y_VLDB = pca_VLDB.map(lambda l: l[1][1]).collect()
    plt.scatter(x=x_VLDB,y=y_VLDB, c=['yellow'], alpha=0.5)
    plt.scatter(x=x_NIPS,y=y_NIPS, c=['blue'], alpha=0.5)
    
list_plot(pca_confernece_scatter)


# In[90]:

# TF-IDF Version for titles

tfidf_rescaled_confernece = tf_idf(cv_conference_result,1000)
pca_conference_tfidf_result,pca_model_conference_tfidf = pca_generic(tfidf_rescaled_confernece,50,"idfFeatures")


# In[94]:

# Plot the eigenvalues of the principal components. Calculate how many components
# are needed to explain 50% of the total variance?

eigenvalues_conference_tfidf = plot_eigenvalues(pca_model_conference_tfidf)


# In[95]:

cumsum_variances(eigenvalues_conference_tfidf)


# We need 19 principal componenets to describe 50% of the principal components

# In[96]:

# Identify which words are important in each of the principal components. To do
# so, take the sum of squares of each of the component vectors to check how they
# are normalized. For each component, then print out the words for which the
# absolute value of the component is larger than 0.20 of the norm.

most_imp_words(pca_model_conference_tfidf,cv_conference_model)


# In[101]:

# Now make a scatter plot of these two principal components, showing the titles from each subset in different colors.
pca_conference_tfidf_scatter,pca_model_conference_tfidf_scatter = pca_generic(tfidf_rescaled_confernece,2,"idfFeatures")
list_plot(pca_conference_tfidf_scatter)


# ### Compare Word Count to TF-IDF
# Here both CountVectorizer and TF-IDF versions are fed with vectors with stopwords removed. When we compare there scatter plots, CountVectorizer with PCA gives more structured scatter plot as compared to TF-IDF. Patterns/Comuunities are relevant in CountVectorizer version more. Also comparing the variance captured by first two componenets count vectors capture approx 14% and Tf-idf captures 8%, therfore distiction in groups if visible in a better way in count vector as compared to tf-idf

# ### Did PCA succeed in uncovering differences between communities.
# Yes PCA was able to uncover differences between academic communities to some extent, but still in both the cases that is word count version and TF-IDF version we see overlapping of communities. This also shows us that PCA is not very good with text semantics since the overlap is because different communities can use the same words in different contexts and that results in overlaps. 
