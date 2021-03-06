
import numpy as np
import pandas as pd
import sys, time, pickle
"""
python get_causal_effect_dict.py THA_topic/check_topic_causal_data_w14h14_from2013_minprob0.05/causal_effect/effect_dict_pw3714_biy1_nocheck_0.05_3.csv THA_topic/check_topic_causal_data_w14h14_from2013_minprob0.05/causal_effect/effect_dict_pw3714_biy1_nocheck_0.05_7.csv THA_topic/check_topic_causal_data_w14h14_from2013_minprob0.05/causal_effect/effect_dict_pw3714_biy1_nocheck_0.05_14.csv THA_w7h7_minday7 0.05
python get_causal_effect_dict.py THA_topic/check_topic_causal_data_w14h14_from2013_minprob0.05/causal_effect/effect_dict_pw3714_biy1_nocheck_0.1_3.csv THA_topic/check_topic_causal_data_w14h14_from2013_minprob0.05/causal_effect/effect_dict_pw3714_biy1_nocheck_0.1_7.csv THA_topic/check_topic_causal_data_w14h14_from2013_minprob0.05/causal_effect/effect_dict_pw3714_biy1_nocheck_0.1_14.csv THA_w7h7_minday7 0.1

python get_causal_effect_dict.py EGY_topic/check_topic_causal_data_w14h14_from2013_minprob0.05/causal_effect/effect_dict_pw3714_biy1_nocheck_0.05_3.csv EGY_topic/check_topic_causal_data_w14h14_from2013_minprob0.05/causal_effect/effect_dict_pw3714_biy1_nocheck_0.05_7.csv EGY_topic/check_topic_causal_data_w14h14_from2013_minprob0.05/causal_effect/effect_dict_pw3714_biy1_nocheck_0.05_14.csv EGY_w7h7_minday7 0.05
python get_causal_effect_dict.py EGY_topic/check_topic_causal_data_w14h14_from2013_minprob0.05/causal_effect/effect_dict_pw3714_biy1_nocheck_0.1_3.csv EGY_topic/check_topic_causal_data_w14h14_from2013_minprob0.05/causal_effect/effect_dict_pw3714_biy1_nocheck_0.1_7.csv EGY_topic/check_topic_causal_data_w14h14_from2013_minprob0.05/causal_effect/effect_dict_pw3714_biy1_nocheck_0.1_14.csv EGY_w7h7_minday7 0.1

python get_causal_effect_dict.py AFG_topic/check_topic_causal_data_w14h14_from2013_minprob0.05/causal_effect/effect_dict_pw3714_biy1_nocheck_0.05_3.csv AFG_topic/check_topic_causal_data_w14h14_from2013_minprob0.05/causal_effect/effect_dict_pw3714_biy1_nocheck_0.05_7.csv AFG_topic/check_topic_causal_data_w14h14_from2013_minprob0.05/causal_effect/effect_dict_pw3714_biy1_nocheck_0.05_14.csv AFG_w7h7_minday7 0.05
python get_causal_effect_dict.py AFG_topic/check_topic_causal_data_w14h14_from2013_minprob0.05/causal_effect/effect_dict_pw3714_biy1_nocheck_0.1_3.csv AFG_topic/check_topic_causal_data_w14h14_from2013_minprob0.05/causal_effect/effect_dict_pw3714_biy1_nocheck_0.1_7.csv AFG_topic/check_topic_causal_data_w14h14_from2013_minprob0.05/causal_effect/effect_dict_pw3714_biy1_nocheck_0.1_14.csv AFG_w7h7_minday7 0.1

python get_causal_effect_dict.py RUS_topic/check_topic_causal_data_w14h14_from2013_minprob0.1/causal_effect/effect_dict_pw3714_biy1_nocheck_0.05_3.csv RUS_topic/check_topic_causal_data_w14h14_from2013_minprob0.1/causal_effect/effect_dict_pw3714_biy1_nocheck_0.05_7.csv RUS_topic/check_topic_causal_data_w14h14_from2013_minprob0.1/causal_effect/effect_dict_pw3714_biy1_nocheck_0.05_14.csv RUS_w7h7_minday10_mindf0.01 0.05
python get_causal_effect_dict.py RUS_topic/check_topic_causal_data_w14h14_from2013_minprob0.1/causal_effect/effect_dict_pw3714_biy1_nocheck_0.1_3.csv RUS_topic/check_topic_causal_data_w14h14_from2013_minprob0.1/causal_effect/effect_dict_pw3714_biy1_nocheck_0.1_7.csv RUS_topic/check_topic_causal_data_w14h14_from2013_minprob0.1/causal_effect/effect_dict_pw3714_biy1_nocheck_0.1_14.csv RUS_w7h7_minday10_mindf0.01 0.1
 


python get_causal_effect_dict.py RUS_2012_60_topic/raw_topic_causal_data_w14h14_from2013_minprob0.15_ngram-1/causal_effect/effect_dict_pw3714_biy1_nocheck_0.05_3.csv RUS_2012_60_topic/raw_topic_causal_data_w14h14_from2013_minprob0.15_ngram-1/causal_effect/effect_dict_pw3714_biy1_nocheck_0.05_7.csv RUS_2012_60_topic/raw_topic_causal_data_w14h14_from2013_minprob0.15_ngram-1/causal_effect/effect_dict_pw3714_biy1_nocheck_0.05_14.csv RUS_w7h7_mind3n10df0.01 0.4 60
python get_causal_effect_dict.py AFG_2012_60_topic/raw_topic_causal_data_w14h14_from2013_minprob0.1_ngram-1/causal_effect/effect_dict_pw3714_biy1_nocheck_0.05_3.csv AFG_2012_60_topic/raw_topic_causal_data_w14h14_from2013_minprob0.1_ngram-1/causal_effect/effect_dict_pw3714_biy1_nocheck_0.05_7.csv AFG_2012_60_topic/raw_topic_causal_data_w14h14_from2013_minprob0.1_ngram-1/causal_effect/effect_dict_pw3714_biy1_nocheck_0.05_14.csv AFG_w7h7_mind3n7df0.01 0.4 60
python get_causal_effect_dict.py EGY_2012_60_topic/raw_topic_causal_data_w14h14_from2013_minprob0.1_ngram-1/causal_effect/effect_dict_pw3714_biy1_nocheck_0.05_3.csv EGY_2012_60_topic/raw_topic_causal_data_w14h14_from2013_minprob0.1_ngram-1/causal_effect/effect_dict_pw3714_biy1_nocheck_0.05_7.csv EGY_2012_60_topic/raw_topic_causal_data_w14h14_from2013_minprob0.1_ngram-1/causal_effect/effect_dict_pw3714_biy1_nocheck_0.05_14.csv EGY_w7h7_mind3n7df0.01 0.4 60
python get_causal_effect_dict.py THA_2012_50_topic/raw_topic_causal_data_w14h14_from2013_minprob0.1_ngram-1/causal_effect/effect_dict_pw3714_biy1_nocheck_0.1_3.csv THA_2012_50_topic/raw_topic_causal_data_w14h14_from2013_minprob0.1_ngram-1/causal_effect/effect_dict_pw3714_biy1_nocheck_0.1_7.csv THA_2012_50_topic/raw_topic_causal_data_w14h14_from2013_minprob0.1_ngram-1/causal_effect/effect_dict_pw3714_biy1_nocheck_0.1_14.csv THA_w7h7_mind3n7df0.01 0.4 50


python get_causal_effect_dict.py RUS_2012_60_topic/raw_topic_causal_data_w14h14_from2013_minprob0.15_ngram-1/causal_effect/effect_dict_pw3714_biy1_nocheck_0.4_3.csv RUS_2012_60_topic/raw_topic_causal_data_w14h14_from2013_minprob0.15_ngram-1/causal_effect/effect_dict_pw3714_biy1_nocheck_0.4_7.csv RUS_2012_60_topic/raw_topic_causal_data_w14h14_from2013_minprob0.15_ngram-1/causal_effect/effect_dict_pw3714_biy1_nocheck_0.4_14.csv RUS_w7h7_mind3n10df0.01 0.4 60
python get_causal_effect_dict.py AFG_2012_60_topic/raw_topic_causal_data_w14h14_from2013_minprob0.1_ngram-1/causal_effect/effect_dict_pw3714_biy1_nocheck_0.4_3.csv AFG_2012_60_topic/raw_topic_causal_data_w14h14_from2013_minprob0.1_ngram-1/causal_effect/effect_dict_pw3714_biy1_nocheck_0.4_7.csv AFG_2012_60_topic/raw_topic_causal_data_w14h14_from2013_minprob0.1_ngram-1/causal_effect/effect_dict_pw3714_biy1_nocheck_0.4_14.csv AFG_w7h7_mind3n7df0.01 0.4 60
python get_causal_effect_dict.py EGY_2012_60_topic/raw_topic_causal_data_w14h14_from2013_minprob0.1_ngram-1/causal_effect/effect_dict_pw3714_biy1_nocheck_0.4_3.csv EGY_2012_60_topic/raw_topic_causal_data_w14h14_from2013_minprob0.1_ngram-1/causal_effect/effect_dict_pw3714_biy1_nocheck_0.4_7.csv EGY_2012_60_topic/raw_topic_causal_data_w14h14_from2013_minprob0.1_ngram-1/causal_effect/effect_dict_pw3714_biy1_nocheck_0.4_14.csv EGY_w7h7_mind3n7df0.01 0.4 60
python get_causal_effect_dict.py THA_2012_50_topic/raw_topic_causal_data_w14h14_from2013_minprob0.1_ngram-1/causal_effect/effect_dict_pw3714_biy1_nocheck_0.4_3.csv THA_2012_50_topic/raw_topic_causal_data_w14h14_from2013_minprob0.1_ngram-1/causal_effect/effect_dict_pw3714_biy1_nocheck_0.4_7.csv THA_2012_50_topic/raw_topic_causal_data_w14h14_from2013_minprob0.1_ngram-1/causal_effect/effect_dict_pw3714_biy1_nocheck_0.4_14.csv THA_w7h7_mind3n7df0.01 0.4 50


python get_causal_effect_dict.py RUS_2012_60_topic/raw_topic_causal_data_w14h14_from2013_minprob0.15_ngram-1/causal_effect/effect_dict_pw3714_biy1_nocheck_1.0_3.csv RUS_2012_60_topic/raw_topic_causal_data_w14h14_from2013_minprob0.15_ngram-1/causal_effect/effect_dict_pw3714_biy1_nocheck_1.0_7.csv RUS_2012_60_topic/raw_topic_causal_data_w14h14_from2013_minprob0.15_ngram-1/causal_effect/effect_dict_pw3714_biy1_nocheck_1.0_14.csv RUS_w7h7_mind3n10df0.01 1.0 60
python get_causal_effect_dict.py AFG_2012_60_topic/raw_topic_causal_data_w14h14_from2013_minprob0.1_ngram-1/causal_effect/effect_dict_pw3714_biy1_nocheck_1.0_3.csv AFG_2012_60_topic/raw_topic_causal_data_w14h14_from2013_minprob0.1_ngram-1/causal_effect/effect_dict_pw3714_biy1_nocheck_1.0_7.csv AFG_2012_60_topic/raw_topic_causal_data_w14h14_from2013_minprob0.1_ngram-1/causal_effect/effect_dict_pw3714_biy1_nocheck_1.0_14.csv AFG_w7h7_mind3n7df0.01 1.0 60
python get_causal_effect_dict.py EGY_2012_60_topic/raw_topic_causal_data_w14h14_from2013_minprob0.1_ngram-1/causal_effect/effect_dict_pw3714_biy1_nocheck_1.0_3.csv EGY_2012_60_topic/raw_topic_causal_data_w14h14_from2013_minprob0.1_ngram-1/causal_effect/effect_dict_pw3714_biy1_nocheck_1.0_7.csv EGY_2012_60_topic/raw_topic_causal_data_w14h14_from2013_minprob0.1_ngram-1/causal_effect/effect_dict_pw3714_biy1_nocheck_1.0_14.csv EGY_w7h7_mind3n7df0.01 1.0 60
python get_causal_effect_dict.py THA_2012_50_topic/raw_topic_causal_data_w14h14_from2013_minprob0.1_ngram-1/causal_effect/effect_dict_pw3714_biy1_nocheck_1.0_3.csv THA_2012_50_topic/raw_topic_causal_data_w14h14_from2013_minprob0.1_ngram-1/causal_effect/effect_dict_pw3714_biy1_nocheck_1.0_7.csv THA_2012_50_topic/raw_topic_causal_data_w14h14_from2013_minprob0.1_ngram-1/causal_effect/effect_dict_pw3714_biy1_nocheck_1.0_14.csv THA_w7h7_mind3n7df0.01 1.0 50


"""
try:
    effect3 = sys.argv[1]
    effect7 = sys.argv[2]
    effect14 = sys.argv[3]
    outpath = sys.argv[4]
    note = sys.argv[5]
    num_topics = int(sys.argv[6])
except:
    print('Usage: effect3, effect7, effect14 path (../data/+...) outpath, note, num_topics ')
    exit()

# THA_topic/check_topic_causal_data_w14h14_from2013_minprob0.05/causal_effect/effect_dict_pw3714_biy1_nocheck_0.05_3.csv
# THA_topic/check_topic_causal_data_w14h14_from2013_minprob0.05/causal_effect/effect_dict_pw3714_biy1_nocheck_0.05_7.csv
# THA_topic/check_topic_causal_data_w14h14_from2013_minprob0.05/causal_effect/effect_dict_pw3714_biy1_nocheck_0.05_14.csv

splitted_date_lists = [
    '2013-01-01','2013-04-01','2013-07-01','2013-10-01',
    '2014-01-01','2014-04-01','2014-07-01','2014-10-01',
    '2015-01-01','2015-04-01','2015-07-01','2015-10-01',
    '2016-01-01','2016-04-01','2016-07-01','2016-10-01',
    '2017-01-01','2017-04-01'
]

causal_file = '../data/'+effect3
causal_df = pd.read_csv(causal_file,sep=',')
causal_df = causal_df.loc[causal_df['event-type']=='protest']
causal_time_dict_3day = {}
for end_date in splitted_date_lists:
        tmp = causal_df.loc[causal_df['end-date']==end_date]
        causal_topic_effect = tmp[['topic-id','effect']].values
        effect_all_topic = np.zeros(num_topics)#[0. for i in range(50)]
        for topic_id, eff in causal_topic_effect:
            effect_all_topic[int(topic_id)] = round(eff,5)
        causal_time_dict_3day[end_date] = effect_all_topic

causal_file = '../data/'+effect7
causal_df = pd.read_csv(causal_file,sep=',')
causal_df = causal_df.loc[causal_df['event-type']=='protest']
causal_time_dict_7day = {}
for end_date in splitted_date_lists:
        tmp = causal_df.loc[causal_df['end-date']==end_date]
        causal_topic_effect = tmp[['topic-id','effect']].values
        effect_all_topic = np.zeros(num_topics)#[0. for i in range(50)]
        for topic_id, eff in causal_topic_effect:
            effect_all_topic[int(topic_id)] = round(eff,5)
        causal_time_dict_7day[end_date] = effect_all_topic

causal_file = '../data/'+effect14
causal_df = pd.read_csv(causal_file,sep=',')
causal_df = causal_df.loc[causal_df['event-type']=='protest']
causal_time_dict_14day = {}
for end_date in splitted_date_lists:
        tmp = causal_df.loc[causal_df['end-date']==end_date]
        causal_topic_effect = tmp[['topic-id','effect']].values
        effect_all_topic = np.zeros(num_topics)#[0. for i in range(50)]
        for topic_id, eff in causal_topic_effect:
            effect_all_topic[int(topic_id)] = round(eff,5)
        causal_time_dict_14day[end_date] = effect_all_topic
causal_time_dict = {}
for k in causal_time_dict_14day:
        v3 = causal_time_dict_3day[k]
        v7 = causal_time_dict_7day[k]
        v14 = causal_time_dict_14day[k]
        v = np.stack((v3,v7,v14),1) # (50,3)
        causal_time_dict[k] = v

print(causal_time_dict.keys(),causal_time_dict)
with open('../data/'+outpath+'/causal_topics_{}.pkl'.format(note),'wb') as f:
    pickle.dump(causal_time_dict,f)