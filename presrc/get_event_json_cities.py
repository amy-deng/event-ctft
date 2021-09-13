import pandas as pd
import numpy as np
# import matplotlib as mpl
# import matplotlib.pyplot as plt
from datetime import date, timedelta
# from matplotlib.pyplot import cm
from scipy.stats import pearsonr,spearmanr
import sys
import json
# from minepy import MINE
# from minepy import pstats, cstats
import time
# %matplotlib inline

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
    start_year = sys.argv[2]
    end_year = sys.argv[3]
    outf = sys.argv[4]
except:
    print("Usage: <country_code> <start_year> <end_year> <out_path>")
    exit()

# country_name = 'RUS'
# country_name = 'GBR'
# country_name = 'TUR'
# country_name = 'PAK'
# country_name = 'IND'

event_file = '~/data/icews/events.sent.new.1991.201703.country/icews_events_{}.json'.format(country_name)
df = pd.read_json(event_file,lines=True)
print('# events = {}'.format(len(df)))
df = df.loc[df['Event Date'] > str(start_year)+'-01-01']
print('# events = {} \t start_year = {}'.format(len(df),start_year))


df2 = df.drop_duplicates(subset=['Country', 'CAMEO Code', 'Event Date', 'Story ID',  'Sentence Number' ])
print('# events = {} \t remove duplicates'.format(len(df2)))
df2.sort_values(by=['Event Date','Event ID'],inplace=True)

# fill data
n_cities = 101
cities = df2['City'].value_counts()[:n_cities].index.tolist()
cities = np.array([v for v in cities if v != ''])
# get lon and lat
lon_lat_dict = {}
for c in cities:
    if c != '':
      lon = df2.loc[df2['City'] == c]['Longitude'].unique()
      lat = df2.loc[df2['City'] == c]['Latitude'].unique()
      # if len(lon) == len(lat) == 1:
      lon_lat_dict[(lon[0],lat[0])] = c
        # else:
        #     print(c)
for k in lon_lat_dict:
    df2.loc[(df2['City'] == '')&(df2['Longitude'] == k[0])&(df2['Latitude'] == k[1]),'City'] = lon_lat_dict[k]

def getRoot(x):
    x = int(x)
    if len(str(x)) == 4: # 1128
        return x // 100
    elif len(str(x)) == 3:
        if x // 10 < 20: # 190
            return x // 10
        else:
            return x // 100
    else:
        return x // 10

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


s_year = int(start_year)
e_year = int(end_year)

start = str(s_year) + '-01-01'
if e_year == 2017:
    end = str(e_year) + '-03-26'
else:
    end = str(e_year) + '-12-31'

# for city in cities:
# outf = 'event.json'
outf = open(outf,'a')
df = df2
df['RootEventCode'] = df['CAMEO Code'].apply(lambda x: getRoot(x) )
for date_i in pd.date_range(start, end, freq='1D'):
    event_date = str(date_i.strftime("%Y-%m-%d"))
    print('event_date',event_date)
    for city in cities:
        filter_events = df.loc[ (df['Event Date'] == event_date ) & (df['City']== city)] 
        if filter_events.empty:
            continue
        else:
            r = dict()
            filter_stories = filter_events['Story ID']
            stories = list(set(list(filter_stories.values)))
            r['story_ids'] = stories
            r['event_date'] = event_date
            r['city'] = city
            r['province'] = filter_events.iloc[0]['Province']
            r['event_ids'] = list(filter_events['Event ID'].values)
            r['event_count'] = filter_events['RootEventCode'].value_counts().to_dict()
            r_json = json.dumps(r, cls=NpEncoder)
            # print(r_json)
            outf.write(r_json)
            outf.write('\n')
        print (time.ctime(),' events in',city, 'processed')
    print (time.ctime(),' events in',event_date, 'processed')
outf.close()

print (time.ctime(),' events processed')
