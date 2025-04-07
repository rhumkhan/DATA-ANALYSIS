#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv("calendar.csv")
df


# In[3]:


df['price'].replace('\$|,','',regex=True,inplace=True)


# In[4]:


df['price']


# In[5]:


df['price'].fillna(0,inplace=True)


# In[6]:


df


# In[7]:


df['price'] = df['price'].astype('float')


# In[8]:


df


# In[9]:


df['date']=pd.to_datetime(df['date'])


# In[10]:


df


# In[11]:


df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['day_name'] = df['date'].dt.day_name()


# In[12]:


df


# In[13]:


# replacing true/false with 1/0 availbility for ease and ploting:
df['available'].replace({'t|T':1,'f|F':0},regex =True,inplace = True)


# In[14]:


df


# In[15]:


df_avail = df.copy()


# In[16]:


available_df = df[df['available'] == 1]
available_df


# In[17]:


sns.boxplot(data=available_df,x = 'month',y = 'price')


# In[37]:


# Group the data by month and count the number of listings
listings = available_df.groupby('month')['listing_id'].count().reset_index()

# Create the bar plot
sns.barplot(data=listings, x='month', y='listing_id')

# Set the labels and title
plt.xlabel("Month")
plt.ylabel("Number of Listings")
plt.title("Distribution of Listings by Month")
plt.show()


# In[38]:


listings


# In[39]:


# EDA ON LISTINGS:


# In[40]:


df1= pd.read_csv("listings.csv")


# In[41]:


df1.info()


# In[43]:


listings_df_miss = pd.DataFrame((df1.isnull().sum())*100/len(df1), columns=['% Missing Values'])
listings_df_miss[listings_df_miss['% Missing Values']>0]


# In[45]:


missing_cols = ['security_deposit','cleaning_fee','review_scores_rating','review_scores_accuracy','review_scores_cleanliness','review_scores_checkin','review_scores_communication','review_scores_location','review_scores_value','reviews_per_month']
df1[missing_cols].info()


# In[47]:


df1['security_deposit'].fillna('$0',inplace=True)
df1['cleaning_fee'].fillna('$0',inplace=True)
df1['security_deposit'] = df1['security_deposit'].str.replace(',','')
df1['security_deposit'] = df1['security_deposit'].str.replace('$','')
df1['security_deposit'] = df1['security_deposit'].astype('float')
df1['cleaning_fee'] = df1['cleaning_fee'].apply(lambda x: ''.join(x.split(',')))
df1['cleaning_fee'] = df1['cleaning_fee'].apply(lambda x: float(x.split('$')[1]))


# In[ ]:


for col in missing_cols:
    df1[col].fillna(0,inplace=True)
df1[missing_cols].sample(5)


# In[48]:


df1['amenities'] = df1['amenities'].apply(lambda x: x[1:-1].split(','))
df1['TV'] = 0
df1['Internet'] = 0
df1['Kitchen'] = 0
df1['Free_parking'] = 0
df1['Washer_dryer'] = 0
df1['Air Conditioning'] = 0
df1['Smoke_detector'] = 0
df1


# In[49]:


for i in range(len(df1)):
    if 'TV' in df1.loc[i,'amenities']:
        df1.loc[i,'TV'] = 1
    if 'Internet' in df1.loc[i,'amenities']:
        df1.loc[i,'Internet'] = 1
    if 'Kitchen' in df1.loc[i,'amenities']:
        df1.loc[i,'Kitchen'] = 1 
    if '"Free Parking on Premises"' in df1.loc[i,'amenities']:
        df1.loc[i,'Free_parking'] = 1
    if 'Washer' in df1.loc[i,'amenities']:
        df1.loc[i,'Washer_dryer'] = 1
    if '"Air Conditioning"' in df1.loc[i,'amenities']:
        df1.loc[i,'Air Conditioning'] = 1
    if '"Smoke Detector"' in df1.loc[i,'amenities']:
        df1.loc[i,'Smoke_detector'] = 1
df1


# In[ ]:


# Fill missing values with '$0'
df1['price'].fillna('$0', inplace=True)
df1['monthly_price'].fillna('$0', inplace=True)
df1['weekly_price'].fillna('$0', inplace=True)

# Remove commas from price columns
df1['price'] = df1['price'].apply(lambda x: ''.join(x.split(',')))
df1['monthly_price'] = df1['monthly_price'].apply(lambda x: ''.join(x.split(',')))
df1['weekly_price'] = df1['weekly_price'].apply(lambda x: ''.join(x.split(',')))

# Convert price columns to float
df1['price'] = df1['price'].apply(lambda x: float(x.split('$')[1]))
df1['monthly_price'] = df1['monthly_price'].apply(lambda x: float(x.split('$')[1]))
df1['weekly_price'] = df1['weekly_price'].apply(lambda x: float(x.split('$')[1]))

# Sample 5 rows to check the changes
df1[['price', 'monthly_price', 'weekly_price']].sample(5)


# In[50]:


# EDA ON REVIEWS:


# In[60]:


df2 = pd.read_csv("reviews.csv")
df2


# In[61]:


#PARSE THE DATE:
df2['date'] = pd.to_datetime(df2['date'])


# In[62]:


df2['Year'] = df2['date'].dt.year
df2['Month'] = df2['date'].dt.month
df2['Day'] =df2['date'].dt.day
df2['day_name'] = df2['date'].dt.day_name()


# In[ ]:


# Define the list of columns from df2 to merge with review_info_df
listing_join_list = ['id', 'price', 'neighbourhood_group_cleansed', 'property_type', 'room_type', 'cleaning_fee',
                     'review_scores_rating', 'review_scores_accuracy', 'review_scores_cleanliness',
                     'review_scores_checkin', 'review_scores_communication', 'review_scores_location',
                     'review_scores_value']

# Group reviews_df by listing_id and count the number of comments (reviews)
review_info_df = pd.DataFrame(df2.groupby('listing_id').count()['comments'].sort_values(ascending=False))

# Merge review_info_df with df2 based on listing_id (reviews_df) and id (df2)
review_info_df = pd.merge(review_info_df, df2[listing_join_list], left_index=True, right_on='id')

# Rename the 'comments' column to 'comment counts'
review_info_df.rename({'comments': 'comment counts'}, axis=1, inplace=True)

# Print the top 10 rows of the merged DataFrame
print(review_info_df.head(10))


# In[ ]:


neg_rating = review_info_df['review_scores_rating'][review_info_df['review_scores_rating'] < 80]
plt.pie([neg_rating.count(), review_info_df.shape[0] - neg_rating.count()], labels=['Negative', 'Positive'],
        autopct='%1.1f%%')
plt.legend()


# In[ ]:


detail_score_rating_l = ['review_scores_accuracy', 'review_scores_cleanliness', 'review_scores_checkin',
                         'review_scores_communication', 'review_scores_location', 'review_scores_value']
mean_score = review_info_df[detail_score_rating_l].mean(axis=0, skipna=True)

data_length = mean_score.shape[0]
angles = np.linspace(0, 2*np.pi, data_length, endpoint=False)
angles = np.concatenate((angles, [angles[0]]))
labels = np.concatenate((mean_score.index.values, [mean_score.index.values[0]]))
scores = np.concatenate((mean_score.values, [mean_score.values[0]]))
fig = plt.figure(figsize=(8, 6), dpi=100)
ax_2_2_a_2 = plt.subplot(111, polar=True)
ax_2_2_a_2.plot(angles, scores, color='g')
ax_2_2_a_2.set_thetagrids(angles*180/np.pi, labels)
ax_2_2_a_2.set_theta_zero_location('N')
ax_2_2_a_2.set_rlim(9, 10)  
ax_2_2_a_2.set_rlabel_position(270)


# In[ ]:


pos_listing_id = review_info_df[review_info_df['review_scores_rating'] >= 80]['id'].values
pos_review_kw_df = reviews_df.loc[reviews_df['listing_id'].isin(pos_listing_id), 'comments']
pos_keyword_vect = CountVectorizer(stop_words='english', min_df=5, max_features=200, decode_error='ignore').\
    fit(pos_review_kw_df)
pos_keyword_ma = pos_keyword_vect.transform(pos_review_kw_df)
print("pos_keyword_matrix:\n{}".format(repr(pos_keyword_ma)))
pos_keyword_dict = pos_keyword_vect.vocabulary_
pos_word_wc = wc.WordCloud(width=1600, height=800)
pos_word_wc.generate_from_frequencies(pos_keyword_dict)
plt.figure(figsize=(20, 8), dpi=400)
plt.tight_layout(pad=0)
plt.imshow(pos_word_wc)  
plt.axis('off') 


# In[ ]:


neg_listing_id = review_info_df[review_info_df['review_scores_rating'] < 80]['id']
neg_review_kw_df = reviews_df.loc[reviews_df['listing_id'].isin(neg_listing_id), 'comments']
neg_keyword_vect = CountVectorizer(stop_words='english', min_df=5, max_features=200, decode_error='ignore').\
    fit(neg_review_kw_df)
neg_keyword_ma = neg_keyword_vect.transform(neg_review_kw_df)
print("neg_keyword_matrix:\n{}".format(repr(neg_keyword_ma)))
neg_keyword_dict = neg_keyword_vect.vocabulary_
neg_word_wc = wc.WordCloud(width=1600, height=800)
neg_word_wc.generate_from_frequencies(neg_keyword_dict)
plt.figure(figsize=(20, 8), dpi=400)
plt.tight_layout(pad=0)
plt.imshow(neg_word_wc)
plt.axis('off')

