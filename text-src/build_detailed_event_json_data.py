import pandas as pd
import numpy as np
import sys, os, json, time
# import pickle
# import glob
# from gensim import corpora, models, similarities
# from gensim.models.ldamulticore import LdaMulticore,LdaModel
# from gensim.test.utils import common_texts
# from gensim.corpora.dictionary import Dictionary
# from gensim.test.utils import common_corpus, common_dictionary

'''
python build_detailed_event_json_data.py THA /home/sdeng/data/icews/detailed_event_json 2010 14 7
'''
try:
    country = sys.argv[1]
    out_path = sys.argv[2]
    # topic_model_name = sys.argv[2] # THA
    start_year = int(sys.argv[3])
    # end_year = int(sys.argv[4])
    window = int(sys.argv[4])
    horizon = int(sys.argv[5])
except:
    print("usage: <country> <out_path `/home/sdeng/data/icews/detailed_event_json`> <start_year> <window> <horizon>")
    exit()

if not os.path.exists(out_path):
    print(out_path, 'not exist')
    exit()

path = "/home/sdeng/data/icews/events.1991.201703.country/icews_events_{}.json".format(country)
df = pd.read_json(path,lines=True)
df.drop(columns=['Longitude', 'Latitude','Source Country','Target Country','Publisher','Target Sectors','Source Sectors','Intensity'],inplace=True)
df = df.drop_duplicates(subset=['Country', 'CAMEO Code', 'Event Date', 'Story ID',  'Sentence Number','Source Name','Target Name' ])
df = df.loc[df['Event Date'] > str(start_year-1)+'-12-01']
print('# cleaned events after {}-12-01 = {}'.format(start_year-1,len(df)))

# def getRoot(x):
#     x = int(x)
#     if len(str(x)) == 4: # 1128
#         return x // 100
#     elif len(str(x)) == 3:
#         if x // 10 <= 20: # 190
#             return x // 10
#         else:
#             return x // 100
#     else:
#         return x // 10

def get_cameo_main(code):
    code = int(code)
    if code < 100:
        return code // 10
    if code // 10 > 20:
        return code // 100
    else:
        return code // 10

df['root'] = df['CAMEO Code'].apply(lambda x: get_cameo_main(x))
print(df.columns)


df_has_city = df.loc[df['City']!='']
print('# events have city'.format(len(df_has_city)))


city_list = df_has_city['City'].unique()

# window = 14
# horizon = 7
outf = "{}/{}_{}_w{}h{}_city.json".format(out_path,country,start_year,window,horizon)
outf = open(outf,'a')

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


for city in city_list:
    df_city = df_has_city.loc[df_has_city['City'] == city]
    date_list = df_city['Event Date'].unique()
    if len(date_list) <= 1:
        continue
    print (time.ctime(),' events in',city, 'processing')
    print('# dates',len(date_list),date_list[:6])
    for day in date_list:
        if day > '2017-03-20' or day < '{}-01-01'.format(start_year):
            continue
        r = dict()
        # get previous 7 days and next 7 days
        prev_dates = list(pd.date_range(end=day, periods=window, closed=None).strftime('%Y-%m-%d'))
#         print('prev_dates',prev_dates)
        story_list = []
        for d in prev_dates:
            df_prev = df_city.loc[df_city['Event Date'] == d]
            storyid = df_prev['Story ID'].unique()
            story_list.append(list(storyid))
        # save news article
        r['story_list'] = story_list
        r['city'] = city
        r['date'] = day # treatment day
        
        next_dates = list(pd.date_range(start=day, periods=horizon+1, closed='right').strftime('%Y-%m-%d'))
        df_next_all = df_city.loc[df_city['Event Date'].isin(next_dates)]
        r['event_count'] = df_next_all['root'].value_counts().to_dict() # next 7 days
        # TODO, better save for each day
#         print('next_dates',next_dates,df_next_all.empty)
        if df_next_all.empty:
            r['event_ids'] = [[] for i in range(horizon)]
        else:
            event_count_list = []
            event_list = []
            for d in next_dates:
                df_next = df_city.loc[df_city['Event Date'] == d]
#                 print(df_next['Event ID'])
                event_list.append(df_next['Event ID'].unique())
                event_count_list.append(df_next['root'].value_counts().to_dict())
            r['event_ids'] = event_list
            r['event_count_list'] = event_count_list
        # print(r)
        r_json = json.dumps(r, cls=NpEncoder)
        # print(r_json)
        outf.write(r_json)
        outf.write('\n')
outf.close()
print (time.ctime(),'done')
