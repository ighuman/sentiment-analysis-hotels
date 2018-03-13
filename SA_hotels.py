
# coding: utf-8


# In[1]:

import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
import pandas
import numpy as np
import math


# In[2]:

#Download required ntlk packages and lib
get_ipython().system(u'pip --quiet install nltk')
import nltk
nltk.download("vader_lexicon")
nltk.download("stopwords")


# In[3]:

from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.sentiment.util import *
from nltk import tokenize
from nltk.corpus import stopwords


# In[4]:

#Read in from pandas
hotelDf = pandas.read_csv('filePath')
hotelDf.columns=['idNum','filePath','hotelName','reviewColumn','ratingScore','groundTruth']


# In[5]:

hotelDf


# In[6]:

#Instantiation
sid = SentimentIntensityAnalyzer()


# In[7]:

stop = set(stopwords.words('english'))


# In[8]:

reviews = hotelDf['reviewColumn'].as_matrix()


# In[9]:


#Count the frequency of words
from collections import Counter
import re
counter = Counter()
for review in reviews:
        counter.update([word.lower() for word in re.findall(r'\w+', review) if word.lower() not in stop and len(word) > 2])


# In[10]:

#Top k word counted by frequency
k = 500
topk = counter.most_common(k)


# In[12]:

pdlist = []
#Assign Vader score to individual review using Vader compound score
for rownum, review in enumerate(reviews):
    ss = sid.polarity_scores(review)
    if ss['compound'] >= 0:
        binaryrating = 'positive'
    else:
        binaryrating = 'negative'
    pdlist.append([review]+[ss['compound']]+ [binaryrating])
    if (rownum % 100 == 1):
            print("processed %d reviews" % (rownum+1))


# In[13]:

reviewDf = pandas.DataFrame(pdlist)
reviewDf.columns = ['reviewCol','vader','vaderTruth']
reviewDf


# In[14]:

#Find out if a particular review has the word from topk list
freqReview = []
for i in range(len(reviewDf)):
    tempCounter = Counter([word for word in re.findall(r'\w+',reviewDf['reviewCol'][i])])
    topkinReview = [1 if tempCounter[word] > 0 else 0 for (word,wordCount) in topk]
    freqReview.append(topkinReview)


# In[15]:


#Prepare freqReviewDf
freqReviewDf = pandas.DataFrame(freqReview)
dfName = []
for c in topk:
    dfName.append(c[0])
freqReviewDf.columns = dfName
freqReviewDf


# In[16]:

finalreviewDf = reviewDf.join(freqReviewDf)


# In[17]:

finaldf = hotelDf[['hotelName','ratingScore','groundTruth']].join(finalreviewDf)


# In[18]:

finaldf


# In[19]:

#We are only intereseted in this three column for overall analysis
itemAnalysisDf = finaldf[['reviewCol','groundTruth','vader']]


# In[20]:

#Extract a list of hotels
hotelNames = finaldf['hotelName'].unique()
hotelNames


# In[21]:

#Rank the hotel by ground truth rating score
hotelRating = []
for hotel in hotelNames:
    itemDf = finaldf.loc[finaldf['hotelName']==hotel]
    hotelRating.append([hotel,itemDf['ratingScore'].mean()])
hotelRatingDfGt = pandas.DataFrame(hotelRating)
hotelRatingDfGt.columns=['hotelName','avgRatingScore']
hotelRatingDfGt.sort_values('avgRatingScore',ascending=0).head(5)


# In[22]:

#Rank the hotel by vader coumpound score
hotelRating = []
for hotel in hotelNames:
    itemDf = finaldf.loc[finaldf['hotelName']==hotel]
    hotelRating.append([hotel,itemDf['vader'].mean()])
hotelRatingDfVd = pandas.DataFrame(hotelRating)
hotelRatingDfVd.columns=['hotelName','avgVaderScore']
hotelRatingDfVd.sort_values('avgVaderScore',ascending=0).head(5)


# In[23]:

#Rank the hotel by ground truth rating score
hotelRating = []
for hotel in hotelNames:
    itemDf = finaldf.loc[finaldf['hotelName']==hotel]
    hotelRating.append([hotel,itemDf['ratingScore'].mean()])
hotelRatingDfGt = pandas.DataFrame(hotelRating)
hotelRatingDfGt.columns=['hotelName','avgRatingScore']
hotelRatingDfGt.sort_values('avgRatingScore',ascending=12).head(5)


# In[24]:

#Rank the hotel by vader coumpound score
hotelRating = []
for hotel in hotelNames:
    itemDf = finaldf.loc[finaldf['hotelName']==hotel]
    hotelRating.append([hotel,itemDf['vader'].mean()])
hotelRatingDfVd = pandas.DataFrame(hotelRating)
hotelRatingDfVd.columns=['hotelName','avgVaderScore']
hotelRatingDfVd.sort_values('avgVaderScore',ascending=12).head(5)


# In[25]:

#We are only intereseted in this three column for overall analysis
itemAnalysisDf = finaldf[['reviewCol','groundTruth','vader','hotelName']]


# In[26]:

#Add possible Stop Words for Hotel Reviews
stop.add('hotel')
stop.add('room')
stop.add('rooms')
stop.add('stay')
stop.add('staff')


# In[27]:

from collections import Counter
import re
#To find out the most frequent word in review when the ground truth is positive
counter = Counter()
for review in itemAnalysisDf.loc[itemAnalysisDf['groundTruth']=='positive']['reviewCol']:
        counter.update([word.lower() for word in re.findall(r'\w+', review) if word.lower() not in stop and len(word) > 2])


# In[28]:

k=50
topkPos = counter.most_common(k)
topkPos


# In[29]:

from collections import Counter
import re
counter = Counter()
#To find out the most frequent word in review when the ground truth is negative
for review in itemAnalysisDf.loc[itemAnalysisDf['groundTruth']=='negative']['reviewCol']:
        counter.update([word.lower() for word in re.findall(r'\w+', review) if word.lower() not in stop and len(word) > 2])


# In[30]:

k=50
topkNeg = counter.most_common(k)
topkNeg


# In[31]:

from collections import Counter
import re
counter = Counter()
#To find out the most frequent word in review when the vader score is positive
for review in itemAnalysisDf.loc[itemAnalysisDf['vader']>0]['reviewCol']:
        counter.update([word.lower() for word in re.findall(r'\w+', review) if word.lower() not in stop and len(word) > 2])
k=50
topk_v = counter.most_common(k)
topk_v


# In[32]:

from collections import Counter
import re
counter = Counter()
#To find out the most frequent word in review when the vader score is negative
for review in itemAnalysisDf.loc[itemAnalysisDf['vader']<0]['reviewCol']:
        counter.update([word.lower() for word in re.findall(r'\w+', review) if word.lower() not in stop and len(word) > 2])
k=50
topk_v_neg = counter.most_common(k)
topk_v_neg


# In[33]:

from collections import Counter
import re
counter = Counter()
#To find out the most frequent word in review with the worst hotel
for review in itemAnalysisDf.loc[itemAnalysisDf['hotelName']=='Motel 6 Toronto Brampton']['reviewCol']:
        counter.update([word.lower() for word in re.findall(r'\w+', review) if word.lower() not in stop and len(word) > 2])
k=50
topk_wh = counter.most_common(k)
topk_wh


# In[34]:

from collections import Counter
import re
counter = Counter()
#To find out the most frequent word in review with the best hotel
for review in itemAnalysisDf.loc[itemAnalysisDf['hotelName']=='Hampton Inn by Hilton Brampton Toronto']['reviewCol']:
        counter.update([word.lower() for word in re.findall(r'\w+', review) if word.lower() not in stop and len(word) > 2])
k=50
topk_bh = counter.most_common(k)
topk_bh



# In[35]:

gtScore_gt = []
for i in range(len(finaldf)):
    if finaldf['ratingScore'][i]>3:
        gtScore_gt.append(1)
    else:
        gtScore_gt.append(0)


# In[36]:

#Calculate mutual information score using scikit lean package
import sklearn
import sklearn.metrics as metrics
miScore = []
for word in topk:
    miScore.append([word[0]]+[metrics.mutual_info_score(gtScore_gt,finaldf[word[0]].as_matrix())])
miScoredf = pandas.DataFrame(miScore).sort_values(1,ascending=0)
miScoredf.columns = ['Word','MI Score']
miScoredf.head(50)


# In[37]:

score_v = []
for i in range(len(finaldf)):
    if finaldf['vader'][i]>0:
        score_v.append(1)
    else:
        score_v.append(0)


# In[38]:

#Calculate muual information score using scikit lean package
import sklearn
import sklearn.metrics as metrics
miScore_v = []
for word in topk:
    miScore_v.append([word[0]]+[metrics.mutual_info_score(score_v,finaldf[word[0]].as_matrix())])
miScoredf_v = pandas.DataFrame(miScore_v).sort_values(1,ascending=0)
miScoredf_v.columns = ['Word','MI Score']
miScoredf_v.head(50)


# In[39]:

score_1_star = []
for i in range(len(finaldf.loc[finaldf['hotelName']=='Motel 6 Toronto Brampton'])):
    if finaldf.loc[finaldf['hotelName']=='Motel 6 Toronto Brampton']['ratingScore'].as_matrix()[i]>3:
        score_1_star.append(1)
    else:
        score_1_star.append(0)


# In[40]:

#Calculate muual information score using scikit lean package
import sklearn
import sklearn.metrics as metrics
miScore_wh = []
for word in topk:
    miScore_wh.append([word[0]]+[metrics.mutual_info_score(score_1_star,finaldf.loc[finaldf['hotelName']=='Motel 6 Toronto Brampton'][word[0]].as_matrix())])
miScoredf_wh = pandas.DataFrame(miScore_wh).sort_values(1,ascending=0)
miScoredf_wh.columns = ['Word','MI Score']
miScoredf_wh.head(50)


# In[41]:

score_1_star_v = []
for i in range(len(finaldf.loc[finaldf['hotelName']=='Motel 6 Toronto Brampton'])):
    if finaldf.loc[finaldf['hotelName']=='Motel 6 Toronto Brampton']['vader'].as_matrix()[i]>0:
        score_1_star_v.append(1)
    else:
        score_1_star_v.append(0)


# In[42]:

#Calculate muual information score using scikit lean package
import sklearn
import sklearn.metrics as metrics
miScore_wh_v = []
for word in topk:
    miScore_wh_v.append([word[0]]+[metrics.mutual_info_score(score_1_star_v,finaldf.loc[finaldf['hotelName']=='Motel 6 Toronto Brampton'][word[0]].as_matrix())])
miScoredf_wh_v = pandas.DataFrame(miScore_wh_v).sort_values(1,ascending=0)
miScoredf_wh_v.columns = ['Word','MI Score']
miScoredf_wh_v.head(50)


# In[43]:

score_3_star = []
for i in range(len(finaldf.loc[finaldf['hotelName']=='Hampton Inn by Hilton Brampton Toronto'])):
    if finaldf.loc[finaldf['hotelName']=='Hampton Inn by Hilton Brampton Toronto']['ratingScore'].as_matrix()[i]>3:
        score_3_star.append(1)
    else:
        score_3_star.append(0)


# In[44]:

#Calculate muual information score using scikit lean package
import sklearn
import sklearn.metrics as metrics
miScore_bh = []
for word in topk:
    miScore_bh.append([word[0]]+[metrics.mutual_info_score(score_3_star,finaldf.loc[finaldf['hotelName']=='Hampton Inn by Hilton Brampton Toronto'][word[0]].as_matrix())])
miScoredf_bh = pandas.DataFrame(miScore_bh).sort_values(1,ascending=0)
miScoredf_bh.columns = ['Word','MI Score']
miScoredf_bh.head(50)


# In[45]:

score_3_star_v = []
for i in range(len(finaldf.loc[finaldf['hotelName']=='Hampton Inn by Hilton Brampton Toronto'])):
    if finaldf.loc[finaldf['hotelName']=='Hampton Inn by Hilton Brampton Toronto']['vader'].as_matrix()[i]>0:
        score_3_star_v.append(1)
    else:
        score_3_star_v.append(0)


# In[46]:

#Calculate muual information score using scikit lean package
import sklearn
import sklearn.metrics as metrics
miScore_bh_v = []
for word in topk:
    miScore_bh_v.append([word[0]]+[metrics.mutual_info_score(score_3_star_v,finaldf.loc[finaldf['hotelName']=='Hampton Inn by Hilton Brampton Toronto'][word[0]].as_matrix())])
miScoredf_bh_v = pandas.DataFrame(miScore_bh_v).sort_values(1,ascending=0)
miScoredf_bh_v.columns = ['Word','MI Score']
miScoredf_bh_v.head(50)



# In[47]:

#Obtain finaldf again for PMI, it should be same as the finaldf before. But just tobe safe, we retrieve it again
freqReview = []
for i in range(len(reviewDf)):
    tempCounter = Counter([word for word in re.findall(r'\w+',reviewDf['reviewCol'][i])])
    topkinReview = [1 if tempCounter[word] > 0 else 0 for (word,wordCount) in topk]
    freqReview.append(topkinReview)
freqReviewDf = pandas.DataFrame(freqReview)
dfName = []
for c in topk:
    dfName.append(c[0])
freqReviewDf.columns = dfName
finalreviewDf = reviewDf.join(freqReviewDf)
finaldf = hotelDf[['hotelName','ratingScore','groundTruth']].join(finalreviewDf)


# In[48]:

def pmiCal(df, x):
    pmilist=[]
    for i in ['positive','negative']:
        for j in [0,1]:
            px = sum(finaldf['groundTruth']==i)/len(df)
            py = sum(finaldf[x]==j)/len(df)
            pxy = len(finaldf[(finaldf['groundTruth']==i) & (finaldf[x]==j)])/len(df)
            if pxy==0:#Log 0 cannot happen
                pmi = math.log10((pxy+0.0001)/(px*py))
            else:
                pmi = math.log10(pxy/(px*py))
            pmilist.append([i]+[j]+[px]+[py]+[pxy]+[pmi])
    pmidf = pandas.DataFrame(pmilist)
    pmidf.columns = ['x','y','px','py','pxy','pmi']
    return pmidf


# In[49]:

def pmiIndivCal(df,x,gt):
    px = sum(finaldf['groundTruth']==gt)/len(df)
    py = sum(finaldf[x]==1)/len(df)
    pxy = len(finaldf[(finaldf['groundTruth']==gt) & (finaldf[x]==1)])/len(df)
    if pxy==0:#Log 0 cannot happen
        pmi = math.log10((pxy+0.0001)/(px*py))
    else:
        pmi = math.log10(pxy/(px*py))
    return pmi


# In[50]:

#Try calculate all the pmi for top k and store them into one pmidf dataframe
pmilist = []
pmiposlist = []
pmineglist = []
for word in topk:
    pmilist.append([word[0]]+[pmiCal(finaldf,word[0])])
    pmiposlist.append([word[0]]+[pmiIndivCal(finaldf,word[0],'positive')])
    pmineglist.append([word[0]]+[pmiIndivCal(finaldf,word[0],'negative')])
pmidf = pandas.DataFrame(pmilist)
pmiposlist = pandas.DataFrame(pmiposlist)
pmineglist = pandas.DataFrame(pmineglist)
pmiposlist.columns = ['word','pmi']
pmineglist.columns = ['word','pmi']
pmidf.columns = ['word','pmi']


# In[51]:

pmidf['pmi'][8]


# In[52]:

#Sorted top pmi words for positive reviews using ground truth
pmiposlist.sort_values('pmi',ascending=0).head(50)


# In[53]:

#Sorted top pmi words for negative reviews for ground truth
pmineglist.sort_values('pmi',ascending=0).head(50)


# In[54]:

def pmiCal_v(df, x):
    pmilist_v=[]
    for i in ['positive','negative']:
        for j in [0,1]:
            px = sum(finaldf['vaderTruth']==i)/len(df)
            py = sum(finaldf[x]==j)/len(df)
            pxy = len(finaldf[(finaldf['vaderTruth']==i) & (finaldf[x]==j)])/len(df)
            if pxy==0:#Log 0 cannot happen
                pmi = math.log10((pxy+0.0001)/(px*py))
            else:
                pmi = math.log10(pxy/(px*py))
            pmilist_v.append([i]+[j]+[px]+[py]+[pxy]+[pmi])
    pmidf_v = pandas.DataFrame(pmilist_v)
    pmidf_v.columns = ['x','y','px','py','pxy','pmi']
    return pmidf_v


# In[55]:

pmiCal_v(finaldf, 'hampton')


# In[56]:

def pmiIndivCal_v(df,x,vt):
    px = sum(finaldf['vaderTruth']==vt)/len(df)
    py = sum(finaldf[x]==1)/len(df)
    pxy = len(finaldf[(finaldf['vaderTruth']==vt) & (finaldf[x]==1)])/len(df)
    if pxy==0:#Log 0 cannot happen
        pmi = math.log10((pxy+0.0001)/(px*py))
    else:
        pmi = math.log10(pxy/(px*py))
    return pmi


# In[57]:

#Try calculate all the pmi for top k and store them into one pmidf dataframe
pmilist_vt = []
pmiposlist_v = []
pmineglist_v = []
for word in topk:
    pmilist_vt.append([word[0]]+[pmiCal_v(finaldf,word[0])])
    pmiposlist_v.append([word[0]]+[pmiIndivCal_v(finaldf,word[0],'positive')])
    pmineglist_v.append([word[0]]+[pmiIndivCal_v(finaldf,word[0],'negative')])
pmidf_v = pandas.DataFrame(pmilist_vt)
pmiposlist_v = pandas.DataFrame(pmiposlist_v)
pmineglist_v = pandas.DataFrame(pmineglist_v)
pmiposlist_v.columns = ['word','pmi']
pmineglist_v.columns = ['word','pmi']
pmidf_v.columns = ['word','pmi']


# In[58]:

#Sorted top pmi words for positive reviews using vader truth
pmiposlist_v.sort_values('pmi',ascending=0).head(50)


# In[59]:

#Sorted top pmi words for negative reviews for vader truth
pmineglist_v.sort_values('pmi',ascending=0).head(50)


# In[60]:

tophoteldf=finaldf.loc[finaldf['hotelName']== 'Hampton Inn by Hilton Brampton Toronto'].reset_index()
tophoteldf


# In[61]:

# pmi for top hotel
def pmiCal_ht_gt(df, x):
    pmilist=[]
    for i in ['positive','negative']:
        for j in [0,1]:
            px = sum(tophoteldf['groundTruth']==i)/len(df)
            py = sum(tophoteldf[x]==j)/len(df)
            pxy = len(tophoteldf[(tophoteldf['groundTruth']==i) & (tophoteldf[x]==j)])/len(df)
            if pxy==0:#Log 0 cannot happen
                pmi = math.log10((pxy+0.0001)/(px*py))
            else:
                pmi = math.log10(pxy/(px*py))
            pmilist.append([i]+[j]+[px]+[py]+[pxy]+[pmi])
    pmidf = pandas.DataFrame(pmilist)
    pmidf.columns = ['x','y','px','py','pxy','pmi']
    return pmidf


# In[62]:

def pmiIndivCal_ht_gt(df,x,gt):
    px = sum(tophoteldf['groundTruth']==gt)/len(df)
    py = sum(tophoteldf[x]==1)/len(df)
    pxy = len(tophoteldf[(tophoteldf['groundTruth']==gt) & (tophoteldf[x]==1)])/len(df)
    if pxy==0:#Log 0 cannot happen
        pmi = math.log10((pxy+0.0001)/(px*py))
    else:
        pmi = math.log10(pxy/(px*py))
    return pmi


# In[63]:

#Try calculate all the pmi for top k and store them into one pmidf dataframe
pmilist_ht_gt = []
pmiposlist_ht_gt = []
pmineglist_ht_gt = []
for word in topk_bh:
    pmilist_ht_gt.append([word[0]]+[pmiCal_ht_gt( tophoteldf,word[0])])
    pmiposlist_ht_gt.append([word[0]]+[pmiIndivCal_ht_gt(tophoteldf,word[0],'positive')])
    pmineglist_ht_gt.append([word[0]]+[pmiIndivCal_ht_gt(tophoteldf,word[0],'negative')])
pmidf = pandas.DataFrame(pmilist)
pmiposlist_ht_gt = pandas.DataFrame(pmiposlist_ht_gt)
pmineglist_ht_gt = pandas.DataFrame(pmineglist_ht_gt)
pmiposlist_ht_gt.columns = ['word','pmi']
pmineglist_ht_gt.columns = ['word','pmi']
pmidf.columns = ['word','pmi']


# In[64]:

#Sorted top pmi words for positive reviews using ground truth
pmiposlist_ht_gt.sort_values('pmi',ascending=0).head(50)


# In[65]:

pmineglist_ht_gt.sort_values('pmi',ascending=0).head(50)


# In[66]:

bottomhoteldf=finaldf.loc[finaldf['hotelName']== 'Motel 6 Toronto Brampton'].reset_index()
bottomhoteldf


# In[67]:

# pmi for bottom hotel
def pmiCal_ht_bt(df, x):
    pmilist=[]
    for i in ['positive','negative']:
        for j in [0,1]:
            px = sum(bottomhoteldf['groundTruth']==i)/len(df)
            py = sum(bottomhoteldf[x]==j)/len(df)
            pxy = len(bottomhoteldf[(bottomhoteldf['groundTruth']==i) & (bottomhoteldf[x]==j)])/len(df)
            if pxy==0:#Log 0 cannot happen
                pmi = math.log10((pxy+0.0001)/(px*py))
            else:
                pmi = math.log10(pxy/(px*py))
            pmilist.append([i]+[j]+[px]+[py]+[pxy]+[pmi])
    pmidf = pandas.DataFrame(pmilist)
    pmidf.columns = ['x','y','px','py','pxy','pmi']
    return pmidf


# In[68]:

def pmiIndivCal_ht_bt(df,x,gt):
    px = sum(bottomhoteldf['groundTruth']==gt)/len(df)
    py = sum(bottomhoteldf[x]==1)/len(df)
    pxy = len(bottomhoteldf[(bottomhoteldf['groundTruth']==gt) & (bottomhoteldf[x]==1)])/len(df)
    if pxy==0:#Log 0 cannot happen
        pmi = math.log10((pxy+0.0001)/(px*py))
    else:
        pmi = math.log10(pxy/(px*py))
    return pmi


# In[70]:

#Try calculate all the pmi for top k and store them into one pmidf dataframe
pmilist_ht_bt = []
pmiposlist_ht_bt = []
pmineglist_ht_bt = []
for word in topk:
    pmilist_ht_bt.append([word[0]]+[pmiCal_ht_bt(bottomhoteldf,word[0])])
    pmiposlist_ht_bt.append([word[0]]+[pmiIndivCal_ht_bt(bottomhoteldf,word[0],'positive')])
    pmineglist_ht_bt.append([word[0]]+[pmiIndivCal_ht_bt(bottomhoteldf,word[0],'negative')])
pmidf = pandas.DataFrame(pmilist)
pmiposlist_ht_bt = pandas.DataFrame(pmiposlist_ht_bt)
pmineglist_ht_bt = pandas.DataFrame(pmineglist_ht_bt)
pmiposlist_ht_bt.columns = ['word','pmi']
pmineglist_ht_bt.columns = ['word','pmi']
pmidf.columns = ['word','pmi']


# In[73]:

#Sorted top pmi words for positive reviews using ground truth
pmiposlist_ht_bt.sort_values('pmi',ascending=0).head(500).as_matrix()


# In[74]:

#Sorted top pmi words for positive reviews using ground truth
pmineglist_ht_bt.sort_values('pmi',ascending=0).head(500).as_matrix()



import matplotlib.pyplot as plt
import numpy as np


# In[76]:

finaldf['ratingScore'].groupby(finaldf['hotelName']).hist()


# In[77]:

finaldf['vader'].groupby(finaldf['hotelName']).hist()
fig = plt.gcf()


# In[79]:

#x axis no of reviews, y axis no of hotels with around those number of reviews. 
finaldf['reviewCol'].groupby(finaldf['hotelName']).count().hist()


# In[80]:

#Plot top 5 side-by-side boxplot for top 5 ground truth rated hotel
tp5gthotel = hotelRatingDfGt.sort_values('avgRatingScore',ascending=0).head(5)
tp5gthotel['hotelName'].as_matrix()


# In[81]:

hampton = finaldf.loc[finaldf['hotelName'] == tp5gthotel['hotelName'].as_matrix()[0]]['ratingScore']
hilton = finaldf.loc[finaldf['hotelName'] == tp5gthotel['hotelName'].as_matrix()[1]]['ratingScore']
fairfield = finaldf.loc[finaldf['hotelName'] == tp5gthotel['hotelName'].as_matrix()[2]]['ratingScore']
courtyard = finaldf.loc[finaldf['hotelName'] == tp5gthotel['hotelName'].as_matrix()[3]]['ratingScore']
daysInn = finaldf.loc[finaldf['hotelName'] == tp5gthotel['hotelName'].as_matrix()[4]]['ratingScore']


# In[82]:

data = [hampton, hilton, fairfield, courtyard, daysInn]
# multiple box plots on one figure
plt.figure()
plt.boxplot(data)
plt.show()


# In[84]:

#Plot top 5 side-by-side boxplot for top 5 vader rated hotel
tp5vdhotel = hotelRatingDfVd.sort_values('avgVaderScore',ascending=0).head(5)
tp5vdhotel['hotelName'].as_matrix()


# In[90]:

hampton_vd = finaldf.loc[finaldf['hotelName'] == tp5vdhotel['hotelName'].as_matrix()[0]]['vader']
hilton_vd = finaldf.loc[finaldf['hotelName'] == tp5vdhotel['hotelName'].as_matrix()[1]]['vader']
fairfield_vd = finaldf.loc[finaldf['hotelName'] == tp5vdhotel['hotelName'].as_matrix()[2]]['vader']
holidayInn_vd = finaldf.loc[finaldf['hotelName'] == tp5vdhotel['hotelName'].as_matrix()[3]]['vader']
comfortInn_vd = finaldf.loc[finaldf['hotelName'] == tp5vdhotel['hotelName'].as_matrix()[4]]['vader']


# In[91]:

data = [hampton_vd, hilton_vd, fairfield_vd, holidayInn_vd, comfortInn_vd]
# multiple box plots on one figure
plt.figure()
plt.boxplot(data)
plt.show()


# In[92]:

print ('The mean of the ground truth for hampton is',(np.mean(hampton)))
print ('The mean of the ground truth for hilton is',(np.mean(hilton)))
print ('The mean of the ground truth for fairfield is',(np.mean(fairfield)))
print ('The mean of the ground truth for courtyard is',(np.mean(courtyard)))
print ('The mean of the ground truth for daysInn is',(np.mean(daysInn)))
print ('The var of the ground truth for hampton is',(np.var(hampton)))
print ('The var of the ground truth for hilton is',(np.var(hilton)))
print ('The var of the ground truth for fairfield is',(np.var(fairfield)))
print ('The var of the ground truth for courtyard is',(np.var(courtyard)))
print ('The var of the ground truth for daysInn is',(np.var(daysInn)))

print ('The mean of the vader truth for hampton is',(np.mean(hampton_vd)))
print ('The mean of the vader truth for hilton is',(np.mean(hilton_vd)))
print ('The mean of the vader truth for fairfield is',(np.mean(fairfield_vd)))
print ('The mean of the vader truth for courtyard is',(np.mean(holidayInn_vd)))
print ('The mean of the vader truth for daysInn is',(np.mean(comfortInn_vd)))
print ('The var of the vader truth for hampton is',(np.var(hampton_vd)))
print ('The var of the vader truth for hilton is',(np.var(hilton_vd)))
print ('The var of the vader truth for fairfield is',(np.var(fairfield_vd)))
print ('The var of the vader truth for courtyard is',(np.var(holidayInn_vd)))
print ('The var of the vader truth for daysInn is',(np.var(comfortInn_vd)))




# In[93]:

y = finaldf['ratingScore'].as_matrix()
x = finaldf['vader'].as_matrix()
plt.plot(x, y,"o")


# In[94]:

y = finaldf['ratingScore'].as_matrix()
x = finaldf['vader'].as_matrix()
plt.plot(x, y,"o")


# In[95]:

y = finaldf['reviewCol'].str.len().as_matrix()
x = finaldf['ratingScore'].as_matrix()
plt.plot(x, y,"o")


# In[96]:

y = finaldf['reviewCol'].str.len().as_matrix()
x = finaldf['vader'].as_matrix()
plt.plot(x, y,"o")


# In[97]:

counts = finaldf['reviewCol'].groupby(finaldf['hotelName']).count().as_matrix()
counts


# In[98]:

ratingGt= finaldf['ratingScore'].groupby(finaldf['hotelName']).mean().as_matrix()
ratingGt


# In[99]:

plt.plot(ratingGt,counts,"o")


# In[100]:

ratingVd= finaldf['vader'].groupby(finaldf['hotelName']).mean().as_matrix()
ratingVd


# In[101]:

plt.plot(ratingVd,counts,"o")


# In[ ]:



