#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import required libraries
import pandas as pd
import requests
from bs4 import BeautifulSoup
import numpy as np


# In[2]:


#link to fetch reviews from
url = "https://www.sitejabber.com/reviews/360digitmg.com"


# In[3]:


# Fetch the content of the URL, parse the HTML content using BeautifulSoup and 'html.parser'
soup = BeautifulSoup(requests.get(url).content,'html.parser')


# In[4]:


# Find all HTML elements with the class 'review__text' and store them in the 'Reviews' list
Reviews =soup.find_all(class_='review__text')


# In[5]:


#view reviews
Reviews


# In[6]:


Review360= [] #Initialize an empty list to store cleaned review text
for i in range (0,len(Reviews)): # Loop through each review element
    Review360.append(Reviews[i].get_text().strip()) #Extract text, remove extra spaces, and add to Review360
Review360 #output the final list


# In[7]:


#create a dataframe
df = pd.DataFrame()


# In[8]:


#add a column reviews and add above reviews in it
df['reviews']= Review360


# In[9]:


#view
df.head(20)


# In[13]:


#install textblob
get_ipython().system('pip install textblob')


# In[14]:


#import libraries
from nltk.corpus import stopwords
from textblob import TextBlob
from textblob import Word


# In[15]:


#lower casing 
df['reviews'] = df['reviews'].apply(lambda x:" ".join(x.lower() for x in x.split()))


# In[16]:


#removing punctuations
df['reviews'] = df['reviews'].str.replace('[^\w\s]',"")


# In[17]:


#view
df['reviews'].head()


# In[19]:


#download stopwords
import nltk
nltk.download('stopwords')


# In[20]:


#removing stopwords
stop = stopwords.words()


# In[21]:


df['reviews'] =df['reviews'].apply(lambda x : str(TextBlob(x).correct()))


# In[23]:


get_ipython().system('pip install vaderSentiment')


# In[24]:


import seaborn as sns
import re
import os
import sys
import ast
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

#function for getting the sentiment
cp = sns.color_palette()
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()


# In[27]:


#Generating Sentiment for all the sentence present in the dataset
emptyline=[] 
for row in df['reviews']:
    
    vs = analyzer.polarity_scores(row)
    emptyline.append(vs)


# In[28]:


#creating new dataframe with sentiments
df_sentiments=pd.DataFrame(emptyline)
df_sentiments.head()


# In[30]:


#Merging the sentiments back to our df dataframe
df_c = pd.concat([df.reset_index(drop=True), df_sentiments], axis=1)
df_c


# In[31]:


#add positive and negative labels
import numpy as np
df_c['sentiment'] = np.where(df_c['compound']>=0, 'Positive', 'Negative')
df_c


# In[32]:


#plotting the negative and positive sentiment count
result =df_c['sentiment'].value_counts()
result.plot(kind='bar', rot=0, color=['Green','Red']);


# In[33]:


df1=df["reviews"]


# In[34]:


df1.to_csv(r"E:\DATA SCIENCE PROJECTS\TEXT MINING\PROJECT 1\360Reviews.csv",index=False, header=False)


# In[38]:


with open(r"E:\DATA SCIENCE PROJECTS\TEXT MINING\PROJECT 1\360Reviews.txt","r") as rc:
    review = rc.read()


# In[42]:


#convert above csv to txt file

# Read the CSV file
df = pd.read_csv(r"E:\DATA SCIENCE PROJECTS\TEXT MINING\PROJECT 1\360Reviews.csv")

# Convert the DataFrame to a string
data_as_string = df.to_string(index=False)

# Write the string to a text file
with open(r"E:\DATA SCIENCE PROJECTS\TEXT MINING\PROJECT 1\360Reviews.txt", "w",encoding="utf-8") as txt_file:
    txt_file.write(data_as_string)


# In[44]:


#open the above created txt file
with open(r"E:\DATA SCIENCE PROJECTS\TEXT MINING\PROJECT 1\360Reviews.txt","r", encoding="utf-8") as rc:
    review = rc.read()


# In[45]:


review = review.split("\n")


# In[46]:


review_string = " ".join(review)


# In[48]:


#install word cloud
get_ipython().system('pip install wordcloud')


# In[49]:


#import libraries
from wordcloud import WordCloud
from wordcloud import STOPWORDS
import matplotlib.pyplot as plt


# In[88]:


#wordcloud for frequently occuring words in the review
wordcloud = WordCloud(background_color ='white').generate(review_string)
plt.figure(figsize=(6,6))
plt.imshow (wordcloud)
plt.axis("off")
plt.show()


# In[51]:


watch_reviews_words = review_string.split(" ")


# In[63]:


# positive words 
#Choose the path for +ve words stored in system 
#one can download positive and negative word file from internet  or 
#you can also create file of your own positive and negative words 

with open(r"E:\DATA SCIENCE PROJECTS\TEXT MINING\PROJECT 1\positive-words.txt","r") as pos:
  poswords = pos.read().split("\n")


# In[93]:


# Positive word cloud
# Choosing the only words which are present in positive words
watch_pos_in_pos = " ".join ([w for w in watch_reviews_words if w in poswords])

wordcloud_pos_in_pos = WordCloud(
                      background_color='White',
                      width=2500,
                      height=1800
                     ).generate(watch_pos_in_pos)
plt.figure(2)
plt.axis("off")
plt.title('Positive Word Cloud')
plt.imshow(wordcloud_pos_in_pos)


# In[52]:


# negative words # Choose the path for +ve words stored in system
with open(r"E:\DATA SCIENCE PROJECTS\TEXT MINING\PROJECT 1\negative-words.txt","r") as neg:
  negwords = neg.read().split("\n")


# In[94]:


# negative word cloud
# Choosing the only words which are present in negwords
watch_neg_in_neg = " ".join ([w for w in watch_reviews_words if w in negwords])

wordcloud_neg_in_neg = WordCloud(
                      background_color='black',
                      width=2500,
                      height=1800
                     ).generate(watch_neg_in_neg)
plt.figure(3)
plt.axis("off")
plt.title('Negative Word Cloud')
plt.imshow(wordcloud_neg_in_neg)


# In[54]:


# wordcloud with bigram
import nltk
text = review_string


# In[55]:


# Remove single quote early since it causes problems with the tokenizer.
text = text.replace("'", "")


# In[57]:


import nltk
nltk.download('punkt')


# In[58]:


tokens = nltk.word_tokenize(text)
text1 = nltk.Text(tokens)


# In[59]:


# Remove extra chars and remove stop words.
text_content = [''.join(re.split("[ .,;:!?‘’``''@#$%^_&*()<>{}~\n\t\\\-]", word)) for word in text1]


# In[60]:


# Create a set of stopwords
stopwords_wc = set(STOPWORDS)


# In[61]:


# Remove stop words
text_content = [word for word in text_content if word not in stopwords_wc]


# In[62]:


# Take only non-empty entries
text_content = [s for s in text_content if len(s) != 0]


# In[66]:


nltk.download('wordnet')


# In[67]:


WNL = nltk.WordNetLemmatizer()
# Best to get the lemmas of each word to reduce the number of similar words
text_content = [WNL.lemmatize(t) for t in text_content]


# In[68]:


nltk_tokens = nltk.word_tokenize(text)  
bigrams_list = list(nltk.bigrams(text_content))
print(bigrams_list)


# In[69]:


dictionary2 = [' '.join(tup) for tup in bigrams_list]
print (dictionary2)


# In[70]:


# Using count vectoriser to view the frequency of bigrams
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(ngram_range=(2, 2))
bag_of_words = vectorizer.fit_transform(dictionary2)
vectorizer.vocabulary_


# In[71]:


sum_words = bag_of_words.sum(axis=0)
words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
print(words_freq[:100])


# In[73]:


# Generating wordcloud
words_dict = dict(words_freq)
WC_height = 1000
WC_width = 1500
WC_max_words = 200
wordCloud = WordCloud(max_words=WC_max_words, height=WC_height, width=WC_width)
wordCloud.generate_from_frequencies(words_dict)
plt.figure(4)
#plt.title('Most frequently occurring bigrams connected by same colour and font size')
plt.imshow(wordCloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[ ]:




