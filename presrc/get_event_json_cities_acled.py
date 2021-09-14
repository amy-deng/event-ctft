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
# path = '~/data/ACLED/country-jul23/2015-01-01-2021-07-23-Yemen.csv'

try:
    path = sys.argv[1]
    # country_name = sys.argv[1]
    # start_year = sys.argv[2]
    # end_year = sys.argv[3]
    outf = sys.argv[2]
    n_city = int(sys.argv[3])
except:
    print("Usage: <file_path> <out_path> <n_city>")
    exit()


country_name = path.split('-')[7][:-4]
print(country_name,'country_name')

filename = path.split('/')[-1]
print('path',path,'filename',filename)
start_year = int(filename.split('-')[0])
start_month = int(filename.split('-')[1])
start_day = int(filename.split('-')[2])
end_year = int(filename.split('-')[3])
end_month = int(filename.split('-')[4])
end_day = int(filename.split('-')[5])
df = pd.read_csv(path,sep=';')
df = df.drop_duplicates(subset=['data_id'], keep='first')
df['event_date'] = pd.to_datetime(df['event_date'])
# print(df.columns)
# print(len(df))



# fill data
n_cities = n_city 
cities = df['admin1'].value_counts()[:n_cities].index.tolist()
 
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
        
def getEventTypeID(x):
    return event_map[x]

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

 
start = str(start_year) + '-' + str(start_month) + '-' + str(start_day)
end = str(end_year) + '-' + str(end_month) + '-' + str(end_day)

outf = open(outf,'a')
sub_event_type = df['sub_event_type'].unique().tolist()
sub_event_type.sort()
# sub_event_type
event_map = dict(zip(sub_event_type,range(1,len(sub_event_type)+1)))

df['EventCode'] = df['sub_event_type'].apply(lambda x: getEventTypeID(x) )
for date_i in pd.date_range(start, end, freq='1D'):
    event_date = str(date_i.strftime("%Y-%m-%d"))
    print('event_date',event_date)
    for city in cities:
        filter_events = df.loc[ (df['event_date'] == event_date ) & (df['admin1']== city)] 
        if filter_events.empty:
            continue
        else:
            r = dict()
            # filter_stories = filter_events['Story ID']
            # stories = list(set(list(filter_stories.values)))
            # r['story_ids'] = stories
            r['event_date'] = event_date
            r['city'] = city # Governorate
            # r['province'] = filter_events.iloc[0]['admin2']
            r['event_ids'] = list(filter_events['data_id'].values)
            r['event_count'] = filter_events['EventCode'].value_counts().to_dict()
            r_json = json.dumps(r, cls=NpEncoder)
            # print(r_json)
            outf.write(r_json)
            outf.write('\n')
        print (time.ctime(),' events in',city, 'processed')
    print (time.ctime(),' events in',event_date, 'processed')
outf.close()

print (time.ctime(),' events processed')
