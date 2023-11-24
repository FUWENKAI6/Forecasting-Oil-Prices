# Exercies 4


import pandas as pd
df = pd.read_csv("/Users/fuwenkai/Documents/U of T/2022 Semester/ECO481/assignment 2/NYT_headlines.csv")
df.head()
# Drop duplicates of the headline column
df.drop_duplicates(subset = "Headlines", inplace = True)


import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Convert to lower case
df['preprocess_text'] = df['Headlines'].str.lower()

# Remove non-alphabetic
import re
df['preprocess_text'] = df['preprocess_text'].apply(lambda x: re.sub(r'[^a-zA-Z ]+', '', x))

# Tokenize
nltk.download('punkt')
from nltk.tokenize import word_tokenize
df['preprocess_text'] = df['preprocess_text'].apply(word_tokenize)

# Remove stop words
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
df['preprocess_text'] = df['preprocess_text'].apply(lambda x: [word for word in x if word not in stop_words])

# Perform Stemming
stemmer = PorterStemmer()
df['preprocess_text'] = df['preprocess_text'].apply(lambda x: [stemmer.stem(word) for word in x])

# Create the vocabulary
covid_words = ['covid', 'coronavirus', 'pandemic', 'virus', 'lockdown', 'quarantine', 'vaccination',
               'coronaviru', 'viru', 'mask', 'sanitizer', 'sanitize', 'vaccin', 'pandem', 'quarantin']

df['covid_words'] = df['preprocess_text'].apply(lambda x: [word for word in x if word in covid_words])


# Combine the different headlines
grouped_df = df.groupby('date').agg({'Headlines': lambda x: ' '.join(x)})


# Preprocess
# Convert to lower case
grouped_df['preprocess_text'] = grouped_df['Headlines'].str.lower()

# Remove non-alphabetic
grouped_df['preprocess_text'] = grouped_df['preprocess_text'].apply(lambda x: re.sub(r'[^a-zA-Z ]+', '', x))

# Tokenize
grouped_df['preprocess_text'] = grouped_df['preprocess_text'].apply(word_tokenize)

# Remove stop words
grouped_df['preprocess_text'] = grouped_df['preprocess_text'].apply(lambda x: [word for word in x if word not in stop_words])

# Perform Stemming
stemmer = PorterStemmer()
grouped_df['preprocess_text'] = grouped_df['preprocess_text'].apply(lambda x: [stemmer.stem(word) for word in x])

# Covert into bag of words

corpus = grouped_df['preprocess_text']

import gensim

dictionary = gensim.corpora.Dictionary(corpus)
corpus_gensim = [dictionary.doc2bow(text) for text in corpus]

# Train model
from gensim.models import LdaMulticore
# Set the number of topics
num_topics = 10

# Train the topic model
lda_model = LdaMulticore(corpus_gensim, num_topics=num_topics, id2word=dictionary, passes=10, workers=2)

# Print the model
for num, topic in lda_model.print_topics(-1):
    print('Topic: {} \nWords: {}'.format(num, topic))

# Display the topics
from wordcloud import WordCloud
import matplotlib.pyplot as plt

for num, topic in lda_model.show_topics(num_topics=num_topics, formatted=False):
    plt.figure()
    plt.imshow(WordCloud(background_color='white').fit_words(dict(topic)))
    plt.axis('off')
    plt.title('Topic #{}'.format(num + 1))
    plt.show()

# Find the optimal number of topics
from gensim.models import CoherenceModel


# Test which one is the best using coherence score
# range from 2-20
t_range = range(2,20,2)
coherence_scores = []
for num in t_range:
    lda_model = LdaMulticore(corpus_gensim, num_topics=num_topics, id2word=dictionary, passes=10, workers=2)
    coherence_model = CoherenceModel(model=lda_model, texts=corpus, dictionary=dictionary, coherence='c_v')
    coherence_score = coherence_model.get_coherence()
    coherence_scores.append(coherence_score)

# Display the coherence score
plt.plot(t_range, coherence_scores, color = "red")
plt.title("The Coherence score related to the number of topcis")
plt.xlabel('The number of topics')
plt.ylabel('The Coherence score')
plt.show()
"""
we can find when the number of topics is 16, the coherence score is the highest, achieve optimal.
"""

# Define articles related to covid
df['covid_count'] = df['preprocess_text'].apply(lambda x: sum(1 for word in covid_words if word in x))

# Calculate the totoal number of articles each day
daily_counts = df.groupby("date").size().reset_index(name='total_count')

# Calculate the covid-related articles each day
covid_counts = df.groupby("date")['covid_count'].sum().reset_index()

# Merge totoal count and covid-related count
merged = pd.merge(daily_counts, covid_counts, on='date', how='left')

# Calculate the uncertainty index
merged["uncert_index"] = merged["covid_count"] / merged["total_count"]

"""
Therefore, the merged dataset contains the covid-related uncertainty index for each date
"""


# Define vocabulary
eco_voc = ["uncertainty", "uncertain", "economic", "economy", "congress", "deficit", "federal reserve", "legislation", "regulation", "White House", "uncertainties", "regulatory","the fed"]

# Define the articles related to economic index
df['eco_count'] = df['preprocess_text'].apply(lambda x: sum(1 for word in eco_voc if word in x))

# Calculate the eco-related articles each day
eco_counts = df.groupby("date")['eco_count'].sum().reset_index()

# Merge totoal count and covid-related count
merged2 = pd.merge(daily_counts, eco_counts, on='date', how='left')

# Calculate the uncertainty index
merged2["uncert_index"] = merged2["eco_count"] / merged2["total_count"]

"""
Therefore, the merged dataset contains the eco-related uncertainty index for each date
"""


"""
The vocabulary used in text classification does not include stemming of words, 
which means that variations of words are not accounted for. As a result, the 
classification is coarse and may not be able to capture the full meaning of 
the text. Additionally, there are some combinations of individual words that 
cannot be detected efficiently, which can lead to inaccurate classifications. 
For example, if the vocabulary does not include the phrase "Federal Reserve", 
but instead only includes the individual words "Federal" and "Reserve", 
the classifier may not be able to detect the phrase as a relevant indicator 
of a economic policy related text. Therefore, it is important to carefully consider 
the vocabulary used in text classification and to continually update and refine 
it to improve the accuracy of the classification.
"""


df2 = pd.read_csv("/Users/fuwenkai/Documents/U of T/2022 Semester/ECO481/assignment 2/SP500.csv")
df2.dtypes
adj_close = df2[["Date","Adj Close**"]]
adj_close["Adj"] = adj_close["Adj Close**"].str.replace(',', '').astype(float)
adj_close = adj_close.sort_values(by = 'Date').reset_index(drop = True)
adj_close["daily_return"] = adj_close["Adj"].pct_change()

"""
The return is in the adj_close dataframe
"""


feb = merged[['date', 'uncert_index']][0:27]
mar = merged[['date', 'uncert_index']][28:]
feb["date"] = pd.to_datetime(feb['date'] + ' 2021', format='%b. %d %Y')
mar["date"] = pd.to_datetime(mar['date'] + ' 2021')
combined = pd.concat([feb, mar],axis = 0)

feb = merged2[['date', 'uncert_index']][0:27]
mar = merged2[['date', 'uncert_index']][28:]
feb["date"] = pd.to_datetime(feb['date'] + ' 2021', format='%b. %d %Y')
mar["date"] = pd.to_datetime(mar['date'] + ' 2021')
combined2 = pd.concat([feb, mar],axis = 0)

merged_both = pd.merge(combined, combined2, on = 'date', how = 'inner').sort_values(by = 'date').reset_index(drop = True)

import seaborn as sns
corr_matrix = merged_both.corr()
sns.scatterplot(data=merged_both, x='uncert_index_x', y='uncert_index_y')

"""
The correlation coefficient of 0.173179 between the Covid uncertainty index and
 the coarse economic policy index suggests that there is a weak positive correlation
 between these two variables. This could mean that as the level of uncertainty 
 related to the Covid-19 pandemic increases, policymakers are responding by 
 implementing more economic policies to mitigate the negative effects on the economy.
"""


# a)
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Create dataset only contain covid
df_covid = df[df['covid_count'] != 0]

analyzer = SentimentIntensityAnalyzer()

# Calculate sentiment score
df_covid['sentiment_scores'] = df_covid['preprocess_text'].apply(lambda x: analyzer.polarity_scores(' '.join(x)))

# Group the sentiment and calculate average
df_sentiment = df_covid.groupby('date')['sentiment_scores'].apply(lambda x: pd.DataFrame(x.tolist()).mean())

df_sentiment.plot(kind='line', figsize=(10, 6), title='Daily Covid-Related Sentiment Index')

# b)
analyzer = SentimentIntensityAnalyzer()

# Calculate sentiment score
df['sentiment_scores'] = df['preprocess_text'].apply(lambda x: analyzer.polarity_scores(' '.join(x)))

# Group the sentiment and calculate average
df_sentiment2 = df.groupby('date')['sentiment_scores'].apply(lambda x: pd.DataFrame(x.tolist()).mean())

df_sentiment2.plot(kind='line', figsize=(10, 6), title='Daily Sentiment Index')


"""
We used text analysis and sentiment analysis techniques to extract insights from the dataset. 
We constructed a vocabulary of COVID-19 related words and used it to build a daily COVID-19 uncertainty index. 
We also analyzed the sentiment of the articles using the Vader sentiment lexicon, both on a daily 
basis and for the entire period of the dataset. Our analysis shows that the sentiment of 
COVID-19 related news articles tends to be more negative than positive.
"""




