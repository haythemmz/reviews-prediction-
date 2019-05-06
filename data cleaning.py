
#%%

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re
import nltk 
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import pickle

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from sklearn.feature_extraction.text import CountVectorizer
import os
#print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.



#%%

reviews=pd.read_csv("Reviews.csv")




reviews.head()


# In[ ]:


reviews.shape


# In[ ]:


def missing_data_function(frame):
        total = frame.isnull().sum().sort_values(ascending=False)
        percent = (frame.isnull().sum()*100 / frame.isnull().count()).sort_values(ascending=False)
        missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
        return missing_data


# In[ ]:


missing_data_function(reviews)


# In[ ]:


reviews=reviews.dropna()


# In[ ]:


reviews.shape


# In[ ]:


reviews.keys()


# In[ ]:


reviews=reviews.sort_values(by='Time', axis=0, ascending=True)


# In[ ]:


reviews.groupby(['UserId','Time']).size().reset_index(name='size').sort_values(by='size', axis=0, ascending=False).head(20)


# In[ ]:


reviews[reviews["UserId"]=="A3TVZM3ZIXG8YW"].head(n=20)


# In[ ]:


reviews=reviews.drop_duplicates(subset=['UserId','Time','Summary', 'Text'], keep='first',inplace = False )


# In[ ]:


reviews.shape


# In[ ]:


reviews[reviews['HelpfulnessNumerator'] >reviews['HelpfulnessDenominator']]


# In[ ]:


reviews=reviews[reviews['HelpfulnessNumerator'] <= reviews['HelpfulnessDenominator']]


# In[ ]:


reviews.shape


# In[ ]:


reviews['Score'].value_counts()


# In[ ]:


pd.value_counts(reviews['Score']).plot.bar()


# In[ ]:


reviews['UserId'].nunique()


# In[ ]:


reviews.shape


# In[ ]:


abbr_dict={
        "isn't":"is not",
        "wasn't":"was not",
        "aren't":"are not",
        "weren't":"were not",
        "can't":"can not",
        "couldn't":"could not",
        "don't":"do not",
        "didn't":"did not",
        "shouldn't":"should not",
        "wouldn't":"would not",
        "doesn't":"does not",
        "haven't":"have not",
        "hasn't":"has not",
        "hadn't":"had not",
        "won't":"will not",
        '["\',\.<>()=*#^:;%µ?|&!-123456789]':""}


# In[ ]:


l=[]
for sentence in (reviews['Text'].values):
    sentence = sentence.lower()                 
    cleanr = re.compile('<.*?>')
    sentence = re.sub(cleanr, ' ', sentence)        
    for j in abbr_dict.keys():
                sentence=re.sub(j,abbr_dict[j],sentence)
    l.append(sentence)


# In[ ]:


print(l[0])


# In[ ]:


vectorizer = CountVectorizer()
counter=vectorizer.fit_transform(reviews['Text'].values)


# In[ ]:


reviews['Text'].values


# In[ ]:


counter.shape


#%%
nltk.download("stopwords")

# In[ ]:


Stopwords = set(stopwords.words('english'))
print(Stopwords)


# In[ ]:


"no" in Stopwords


# In[ ]:


"not" in Stopwords


# In[ ]:


#Stopwords=Stopwords.remove('not')
#Stopwords=Stopwords.remove('no')
#print(Stopwords)


# In[ ]:


Stopwords.remove('not')


# In[ ]:


Stopwords.remove('no')


# In[ ]:


def preprocessing(l,stemming):
    snow = nltk.stem.SnowballStemmer('english')
    a=[]
    for j in l :
        if stemming==True:
            a.append([snow.stem(word) for word in j.split() if word not in Stopwords])
        else:
            a.append([word for word in j.split() if word not in Stopwords])        
    return a    
    


# In[ ]:


a=preprocessing(l,stemming=True)


# In[ ]:


print(a[0])


# In[ ]:


def to_sentence(l):
    sentence = []
    for row in l:
        sequ = ''
        for word in row:
            sequ = sequ + ' ' + word
        sentence.append(" ".join(sequ.split()))
    return sentence


# In[ ]:


sentences=to_sentence(a)


# In[ ]:


sentences[0]

#%%
print()
# In[ ]:


reviews['preprocessed_text']=sentences


# In[ ]:


without_stemming=preprocessing(l,stemming=False)


# In[ ]:


reviews['preprocessed_text_without_stemming']=without_stemming


# In[ ]:


word2vect=Word2Vec(without_stemming,min_count=8,size=40, workers=4) 


# In[ ]:


len(list(word2vect.wv.vocab))


# In[ ]:


word2vect.wv.most_similar('love')


# In[ ]:


sent_vectors=[]
for j in without_stemming :
    s=np.zeros(40)
    cnt_words=0
    for k in j: 
        try:
            vec = word2vect.wv[k]
            s += vec
            cnt_words += 1
        except:
            pass
    sent_vectors.append(s/cnt_words)
print(len(sent_vectors))
print(len(sent_vectors[0]))  


# In[ ]:


def n_gram(l,grams):
    count_vect = CountVectorizer(ngram_range=(1,grams))
    n_grams = count_vect.fit_transform(l)
    return n_grams


# In[ ]:


reviews['preprocessed_text']=sentences


# In[ ]:


_2_grams = n_gram(sentences,2) 
reviews["_2_grams"]=_2_grams


# In[ ]:


wordcloud = WordCloud().generate(text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer         
tf_idf = TfidfVectorizer(max_features=5000)
reviews['after_tf_idf'] = tf_idf.fit_transform(reviews['preprocessed_text'])

