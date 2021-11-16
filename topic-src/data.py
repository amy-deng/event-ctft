from typing import List
import numpy as np
import pickle
import torch
from torch.utils import data
# import utils
import pickle
import pandas as pd
import collections

# class StaticGraphData(object):
#     def __init__(self, args):
#         self.dataset = args.dataset # THA_w7h7_minday3
#         # self.model = args.model
#         self.cuda = args.cuda
#         self.window = args.window
#         self.horizon = args.horizon
#         # self.pred_window = args.pred_window
#         with open('{}/{}/data_static_2015-01-01_tt85_ww10.pkl'.format(args.data_path, self.dataset),'rb') as f:
#             data_dict = pickle.load(f)

#         date = data_dict['date'] # list
#         city = data_dict['city'] # list
#         y_data = data_dict['y'] # tensor 
        # g_data = data_dict['graphs_list'] # dgl

# RUS
# Counter({'Moscow': 1173, 'Moscow Kremlin': 918, 'St. Petersburg': 251, 'Sochi': 229, 
# 'Groznyy': 45, 'Vladivostok': 36, 'Ufa': 32, 'Volgograd': 24, 'Rostov': 21, 'Kaliningrad': 17, 
# 'Rostov-na-Donu': 16, 'Red Square': 14, 'Lugansk': 13, 'Sana': 12, 'Yekaterinburg': 12, 
# 'Pskov': 7, 'Kirov': 7, "Vasil'yevka": 6, 'Nalchik Airport': 6, 'Bor': 5, 'Kubinka': 5, 
# 'Irkutsk': 5, 'Khabarovsk': 4, 'Valday': 4, 'Sputnik': 4, 'Samara': 4, 'Novoselitskoye': 4,
#  'Samarkand': 4, 'Voronezh': 3, 'Bolotnaya': 3, 'Slavyanka': 3, 'Tatarstan': 3, 'Derbent': 3, 
# 'Vnukovo International Airport': 2, 'Makhachkala': 2, 'Lada': 2, 'Yuzhno-Sakhalinsk': 2, 
# 'Belgorod': 2, 'Nizhny Novgorod': 2, 'Kostroma': 2, 'Buynaksk': 2, 'Krasnogorsk': 2, 'Domodedovo': 2, 'Balashikha': 2, 'Tver': 1, 'Rada': 1, 'Gukovo': 1, 'Novorossiysk': 1, 'Omsk': 1, 'Yaroslavl': 1, 'Nefteyugansk': 1, 'Krasnaya Polyana': 1, 'Lefortovo': 1})

# EGY
# Counter({'Cairo': 1141, 'Alexandria': 117, 'Arish': 109, 'Giza': 93, 'Sharm ash Shaykh': 58, 
# 'Suez': 42, 'Tahrir Square': 34, 'Al Minya': 31, 'Luxor': 18, 'Cairo University': 12, 'Suez Canal': 12, 
# 'Aswan': 9, 'Al Mansurah': 9, 'Port Said': 9, 'Ismailia': 7, 'Al Karnak': 7, 'Al Ghardaqah': 7, 'Al Fayyum': 5, 'Taba': 5, 'Sohag': 5, 'Madinat an Nasr': 3, 'Heliopolis': 3, 'Saqqarah': 3, 'Halwan': 3, 'Shubra al Khaymah': 2, 'Damietta': 2, 'Tanda': 1})     
class StaticGraphData(data.Dataset):
      def __init__(self, path, dataset, datafiles, horizon, causalfiles=''):
            # data, times = utils.load_quadruples(path + dataset, set_name + '.txt')
            datafile_list = datafiles.split(',')
            datafile_list.sort()
            y_data = []
            g_data = []
            t_data = []
            for datafile in datafile_list:
                  with open('{}/{}/{}.pkl'.format(path, dataset,datafile),'rb') as f:
                        # with open('{}/{}/data_static_2012-01-01_2013-01-01_tt85_ww10.pkl'.format(path, dataset),'rb') as f:
                        data_dict = pickle.load(f)
                  y_data.append(data_dict['y'])
                  g_data += data_dict['graphs_list']
                  t_data +=  data_dict['date']
            # times = torch.from_numpy(times)
            # y_data = data_dict['y'] # tensor 
            # g_data = data_dict['graphs_list'] # dgl
            y_data = torch.cat(y_data,dim=0)
            # print(y_data.shape,'y_data',y_data)
            y_data = y_data[:,:horizon].sum(-1)
            y_data = torch.where(y_data > 0,1.,0.)
            self.len = len(y_data)
            self.y_data = y_data
            self.g_data = g_data
            self.t_data = t_data
            print(len(self.g_data),'self.g_data', 'self.y_data',self.y_data.shape)
            print('positive',y_data.mean()) 
            # '''
            #load causal
            splitted_date_lists = ['2010-07-01',
            '2011-01-01','2011-07-01','2012-01-01','2012-07-01','2013-01-01','2013-07-01',
            '2014-01-01','2014-07-01','2015-01-01','2015-07-01','2016-01-01','2016-07-01',
            '2017-01-01','2017-07-01'
            ]
            self.splitted_date_lists = splitted_date_lists
            if causalfiles != '':
                  causalfile_list = causalfiles.split(',')
                  causalfile_list.sort()
                  df_list = []
                  for i in range(len(causalfile_list)):
                        causal_file = causalfile_list[i]
                        df = pd.read_csv(causal_file+'.csv',sep=',')
                        df = df.loc[df['event-type']=='protest']
                        df_list.append(df)
                  causal_time_dict = {}
                  for end_date in splitted_date_lists:
                        effect_list = []
                        for i in range(len(df_list)):
                              df = df_list[i]
                              tmp = df.loc[df['end-date']==end_date]
                              causal_topic_effect = tmp[['topic-id','effect']].values
                              effect_all_topic = np.zeros(50)#[0. for i in range(50)]
                              for topic_id, eff in causal_topic_effect:
                                    effect_all_topic[int(topic_id)] = round(eff,5)
                              # causal_time_dict_3day[end_date] = effect_all_topic
                              effect_list.append(effect_all_topic)
                        v = np.stack(effect_list,1) # (50,3)
                        causal_time_dict[end_date] = v
                  self.causal_time_dict = causal_time_dict
            else:
                  self.causal_time_dict = {}
            # print('causal_time_dict',self.causal_time_dict)
            # causal_file = '../data/THA_topic/check_topic_causal_data_w7h14/causal_effect/effect_dict_pw3_biy1_nocheck_0.05.csv'
            # causal_df = pd.read_csv(causal_file,sep=',')
            # causal_df = causal_df.loc[causal_df['event-type']=='protest']
            # causal_time_dict_3day = {}
            # for end_date in splitted_date_lists:
            #       tmp = causal_df.loc[causal_df['end-date']==end_date]
            #       causal_topic_effect = tmp[['topic-id','effect']].values
            #       effect_all_topic = np.zeros(50)#[0. for i in range(50)]
            #       for topic_id, eff in causal_topic_effect:
            #             effect_all_topic[int(topic_id)] = round(eff,5)
            #       causal_time_dict_3day[end_date] = effect_all_topic

            # causal_file = '../data/THA_topic/check_topic_causal_data_w7h14/causal_effect/effect_dict_pw7_biy1_nocheck_0.05.csv'
            # causal_df = pd.read_csv(causal_file,sep=',')
            # causal_df = causal_df.loc[causal_df['event-type']=='protest']
            # causal_time_dict_7day = {}
            # for end_date in splitted_date_lists:
            #       tmp = causal_df.loc[causal_df['end-date']==end_date]
            #       causal_topic_effect = tmp[['topic-id','effect']].values
            #       effect_all_topic = np.zeros(50)#[0. for i in range(50)]
            #       for topic_id, eff in causal_topic_effect:
            #             effect_all_topic[int(topic_id)] = round(eff,5)
            #       causal_time_dict_7day[end_date] = effect_all_topic

            # causal_file = '../data/THA_topic/check_topic_causal_data_w7h14/causal_effect/effect_dict_pw14_biy1_nocheck_0.05.csv'
            # causal_df = pd.read_csv(causal_file,sep=',')
            # causal_df = causal_df.loc[causal_df['event-type']=='protest']
            # causal_time_dict_14day = {}
            # for end_date in splitted_date_lists:
            #       tmp = causal_df.loc[causal_df['end-date']==end_date]
            #       causal_topic_effect = tmp[['topic-id','effect']].values
            #       effect_all_topic = np.zeros(50)#[0. for i in range(50)]
            #       for topic_id, eff in causal_topic_effect:
            #             effect_all_topic[int(topic_id)] = round(eff,5)
            #       causal_time_dict_14day[end_date] = effect_all_topic
            # causal_time_dict = {}
            # for k in causal_time_dict_14day:
            #       v3 = causal_time_dict_3day[k]
            #       v7 = causal_time_dict_7day[k]
            #       v14 = causal_time_dict_14day[k]
            #       v = np.stack((v3,v7,v14),1) # (50,3)
            #       causal_time_dict[k] = v
            
            # '''
      def __len__(self):
            return self.len

      def __getitem__(self, index):
            if self.causal_time_dict == {}:
                  return self.g_data[index], self.y_data[index]

            g = self.g_data[index]
            date = self.t_data[index]
            for end_date in self.splitted_date_lists: # check date in which range
                  if date < end_date:
                        cur_end_date = end_date
                        break
            causal_weight = self.causal_time_dict[cur_end_date]
            causal_weight_tensor = torch.from_numpy(causal_weight)#.to_sparse()
            if isinstance(g, list):
                  for i in range(len(g)):
                        g[i].nodes['topic'].data['effect'] = causal_weight_tensor[g[i].nodes('topic').numpy()].to_sparse()
            else:
                  g.nodes['topic'].data['effect'] = causal_weight_tensor[g.nodes('topic').numpy()].to_sparse()
            return g, self.y_data[index]

class StaticGraphDataNew(data.Dataset):
      def __init__(self, path, dataset, datafiles, horizon, causalfiles=''):
            # data, times = utils.load_quadruples(path + dataset, set_name + '.txt')
            datafile_list = datafiles.split(',')
            datafile_list.sort()
            y_data = []
            g_data = []
            t_data = []
            for datafile in datafile_list:
                  with open('{}/{}/{}.pkl'.format(path, dataset,datafile),'rb') as f:
                        # with open('{}/{}/data_static_2012-01-01_2013-01-01_tt85_ww10.pkl'.format(path, dataset),'rb') as f:
                        graph_list = pickle.load(f)
                  g_data += graph_list
                  
                  tmp = datafile.split('_')
                  tmp[0] = 'attr'
                  attr_file = '_'.join(tmp)
                  with open('{}/{}/{}.pkl'.format(path, dataset, attr_file),'rb') as f:
                        # with open('{}/{}/data_static_2012-01-01_2013-01-01_tt85_ww10.pkl'.format(path, dataset),'rb') as f:
                        attr_dict = pickle.load(f)
                  y_data.append(attr_dict['y'])
                  t_data +=  attr_dict['date']
            # times = torch.from_numpy(times)
            # y_data = data_dict['y'] # tensor 
            # g_data = data_dict['graphs_list'] # dgl
            y_data = torch.cat(y_data,dim=0)
            # print(y_data.shape,'y_data',y_data)
            y_data = y_data[:,:horizon].sum(-1)
            y_data = torch.where(y_data > 0,1.,0.)
            self.len = len(y_data)
            self.y_data = y_data
            self.g_data = g_data
            self.t_data = t_data
            print(len(self.g_data),'self.g_data', 'self.y_data',self.y_data.shape)
            print('positive',y_data.mean()) 
            # '''
            #load causal
            splitted_date_lists = ['2010-07-01',
            '2011-01-01','2011-07-01','2012-01-01','2012-07-01','2013-01-01','2013-07-01',
            '2014-01-01','2014-07-01','2015-01-01','2015-07-01','2016-01-01','2016-07-01',
            '2017-01-01','2017-07-01'
            ]
            self.splitted_date_lists = splitted_date_lists
            if causalfiles != '':
                  causalfile_list = causalfiles.split(',')
                  causalfile_list.sort()
                  df_list = []
                  for i in range(len(causalfile_list)):
                        causal_file = causalfile_list[i]
                        df = pd.read_csv(causal_file+'.csv',sep=',')
                        df = df.loc[df['event-type']=='protest']
                        df_list.append(df)
                  causal_time_dict = {}
                  for end_date in splitted_date_lists:
                        effect_list = []
                        for i in range(len(df_list)):
                              df = df_list[i]
                              tmp = df.loc[df['end-date']==end_date]
                              causal_topic_effect = tmp[['topic-id','effect']].values
                              effect_all_topic = np.zeros(50)#[0. for i in range(50)]
                              for topic_id, eff in causal_topic_effect:
                                    effect_all_topic[int(topic_id)] = round(eff,5)
                              # causal_time_dict_3day[end_date] = effect_all_topic
                              effect_list.append(effect_all_topic)
                        v = np.stack(effect_list,1) # (50,3)
                        causal_time_dict[end_date] = v
                  self.causal_time_dict = causal_time_dict
            else:
                  self.causal_time_dict = {}
            # print('causal_time_dict',self.causal_time_dict)
            # causal_file = '../data/THA_topic/check_topic_causal_data_w7h14/causal_effect/effect_dict_pw3_biy1_nocheck_0.05.csv'
            # causal_df = pd.read_csv(causal_file,sep=',')
            # causal_df = causal_df.loc[causal_df['event-type']=='protest']
            # causal_time_dict_3day = {}
            # for end_date in splitted_date_lists:
            #       tmp = causal_df.loc[causal_df['end-date']==end_date]
            #       causal_topic_effect = tmp[['topic-id','effect']].values
            #       effect_all_topic = np.zeros(50)#[0. for i in range(50)]
            #       for topic_id, eff in causal_topic_effect:
            #             effect_all_topic[int(topic_id)] = round(eff,5)
            #       causal_time_dict_3day[end_date] = effect_all_topic

            # causal_file = '../data/THA_topic/check_topic_causal_data_w7h14/causal_effect/effect_dict_pw7_biy1_nocheck_0.05.csv'
            # causal_df = pd.read_csv(causal_file,sep=',')
            # causal_df = causal_df.loc[causal_df['event-type']=='protest']
            # causal_time_dict_7day = {}
            # for end_date in splitted_date_lists:
            #       tmp = causal_df.loc[causal_df['end-date']==end_date]
            #       causal_topic_effect = tmp[['topic-id','effect']].values
            #       effect_all_topic = np.zeros(50)#[0. for i in range(50)]
            #       for topic_id, eff in causal_topic_effect:
            #             effect_all_topic[int(topic_id)] = round(eff,5)
            #       causal_time_dict_7day[end_date] = effect_all_topic

            # causal_file = '../data/THA_topic/check_topic_causal_data_w7h14/causal_effect/effect_dict_pw14_biy1_nocheck_0.05.csv'
            # causal_df = pd.read_csv(causal_file,sep=',')
            # causal_df = causal_df.loc[causal_df['event-type']=='protest']
            # causal_time_dict_14day = {}
            # for end_date in splitted_date_lists:
            #       tmp = causal_df.loc[causal_df['end-date']==end_date]
            #       causal_topic_effect = tmp[['topic-id','effect']].values
            #       effect_all_topic = np.zeros(50)#[0. for i in range(50)]
            #       for topic_id, eff in causal_topic_effect:
            #             effect_all_topic[int(topic_id)] = round(eff,5)
            #       causal_time_dict_14day[end_date] = effect_all_topic
            # causal_time_dict = {}
            # for k in causal_time_dict_14day:
            #       v3 = causal_time_dict_3day[k]
            #       v7 = causal_time_dict_7day[k]
            #       v14 = causal_time_dict_14day[k]
            #       v = np.stack((v3,v7,v14),1) # (50,3)
            #       causal_time_dict[k] = v
            
            # '''
      def __len__(self):
            return self.len

      def __getitem__(self, index):
            if self.causal_time_dict == {}:
                  return self.g_data[index], self.y_data[index]

            g = self.g_data[index]
            date = self.t_data[index]
            for end_date in self.splitted_date_lists: # check date in which range
                  if date < end_date:
                        cur_end_date = end_date
                        break
            causal_weight = self.causal_time_dict[cur_end_date]
            causal_weight_tensor = torch.from_numpy(causal_weight)#.to_sparse()
            if isinstance(g, list):
                  for i in range(len(g)):
                        g[i].nodes['topic'].data['effect'] = causal_weight_tensor[g[i].nodes('topic').numpy()].to_sparse()
            else:
                  g.nodes['topic'].data['effect'] = causal_weight_tensor[g.nodes('topic').numpy()].to_sparse()
            return g, self.y_data[index]



def collate_2(batch):
    g_data = [item[0] for item in batch]
    y_data = [item[1] for item in batch]
    return [g_data, y_data]