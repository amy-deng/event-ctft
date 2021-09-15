import pandas as pd
import numpy as np
# import matplotlib as mpl
# import matplotlib.pyplot as plt
from datetime import date, timedelta
# from matplotlib.pyplot import cm
from scipy.stats import pearsonr,spearmanr
import sys, time, json, re, itertools, pickle
# from minepy import MINE
# from minepy import pstats, cstats
# %matplotlib inline
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
'''
generate dict for each city
{
  "story_ids" : [
      18214139,
      18211903
    ]
  "date": "2013-01-28",
  "city": "Bangkok",
  "province": "Krung Thep Mahanakhon",
  "event_ids": [
    19284204
  ],
  "event_count": {
    "11": 1
  }
}
'''

try:
    country_name = sys.argv[1]
    json_path = sys.argv[2]
    # outf = sys.argv[4]
    n_city = int(sys.argv[3])
except:
    print("Usage: <country_code>  <json_path> <n_city>")
    exit()

# country_name = 'RUS'
# country_name = 'GBR'
# country_name = 'TUR'
# country_name = 'PAK'
# country_name = 'IND'

# event_file = '~/data/icews/events.sent.new.1991.201703.country/icews_events_{}.json'.format(country_name)
# df = pd.read_json(event_file,lines=True)
# print('# events = {}'.format(len(df)))
# df = df.loc[df['Event Date'] > '2010-01-01']
# print('# events = {} \t start_year = {}'.format(len(df),'2010'))


# df2 = df.drop_duplicates(subset=['Country', 'CAMEO Code', 'Event Date', 'Story ID',  'Sentence Number' ])
# print('# events = {} \t remove duplicates'.format(len(df2)))
# df2.sort_values(by=['Event Date','Event ID'],inplace=True)
# df = df2

cities = open('../data/{}/cities.txt'.format(country_name)).read().splitlines()
print(len(cities),cities)

s_df = pd.read_json("/home/sdeng/data/icews/news.1991.201703.country/icews_news_{}.json".format(country_name), lines=True)

event_df = pd.read_json(json_path, lines=True)


# def text_tokenize(text): # list of list
#     lists = []
#     text = re.sub(r"''", "\\n",text) 
#     lists += text.split("\\n")
#     return lists

# def text_tokenize_list(texts): # list of list
#     l = []
#     for t in texts:
#         l += text_tokenize(t)
#     return l

# def get_vector_from_text(sent_list):
#     vectorizer = TfidfVectorizer()
#     X = vectorizer.fit_transform(sent_list)
#     print(X.shape)
#     pass

# def get_tfidf_transformer(event_df,s_df):
#     story_ids = event_df['story_ids'].values.tolist()
#     story_ids = list(set(itertools.chain(*story_ids)))
#     print(len(story_ids),'story_ids')
#     texts_all = list(s_df.loc[s_df['StoryID'].isin(story_ids)]['Text'])
#     print(len(texts_all),'texts_all' )
#     sent_list = text_tokenize_list(texts_all)
#     print(len(sent_list),'sent_list' )
#     vectorizer = TfidfVectorizer()
#     vectorizer = vectorizer.fit(sent_list)
#     return vectorizer

# vectorizer = get_tfidf_transformer(event_df,s_df)

start = '2010-01-01'
end = '2017-03-26'
# target_city = cities[0]
print('from {} to {}'.format(start,end))
n_days = 0
text_all_city = []
for city in cities:
    text_city = []
    for date_i in pd.date_range(start, end, freq='1D'):
        event_date = str(date_i.strftime("%Y-%m-%d"))
        #data in target
        filter_events = event_df.loc[ (event_df['event_date'] == event_date ) & (event_df['city']== city)]
        if filter_events.empty:
            text = ''
        else:
            story_ids = filter_events['story_ids'].values.tolist()
            story_ids = list(set(itertools.chain(*story_ids)))
            texts_all = list(s_df.loc[s_df['StoryID'].isin(story_ids)]['Text'])
            text = ' '.join(texts_all)
        text_city.append(text)
    print(len(text_city),city)
    n_days = len(text_city)
    text_all_city.append(text_city)
 
text_all_city_flatten = list(set(itertools.chain(*text_all_city)))
print(len(text_all_city_flatten))
vectorizer = TfidfVectorizer()
vectorizer = vectorizer.fit(text_all_city_flatten)
vectors = [] 
for i in range(len(text_all_city)):
    tmp = vectorizer.transform(text_all_city[i])
    # print(tmp.shape)
    vectors.append(tmp) # (#day,#feat)
print('get tfidf features for each city')

all_days = []
for i in range(n_days):
    target = vectors[0][i]
    similarity_cities = []
    for j in range(0,len(cities)):
        other = vectors[j][i]
        v = cosine_similarity(target, other )
        similarity_cities.append(round(v[0][0],4))
    all_days.append(similarity_cities)

all_days = np.array(all_days)
print(all_days.shape)
with open('../data/{}/propensity.pkl'.format(country_name),'wb') as f:
    pickle.dump(all_days,f)
print('propensity saved')
# text_all_city = []
# for city in cities:
#     print(city)
#     text_city = []
#     for date_i in pd.date_range(start, end, freq='1D'):
#         event_date = str(date_i.strftime("%Y-%m-%d"))
#         #data in target
#         filter_events = df.loc[ (df['event_date'] == event_date ) & (df['city']== city)]
#         if filter_events.empty:
#             text = []
#         else:
#             story_ids = filter_events['story_ids'].values.tolist()
#             story_ids = list(set(itertools.chain(*story_ids)))
#             texts_all = list(s_df.loc[s_df['StoryID'].isin(story_ids)]['Text'])
#             text = ' '.join(texts_all)
#         text_city.append(text)
#     text_all_city.append(text_city)