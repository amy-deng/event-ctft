import numpy as np
import pickle
import torch
from torch.utils import data
import numpy as np
# import utils
import pickle
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

        
class StaticGraphData(data.Dataset):
      def __init__(self, path, dataset, set_name='train'):
            # data, times = utils.load_quadruples(path + dataset, set_name + '.txt')
            with open('{}/{}/data_static_2012-01-01_2012-01-11_tt85_ww10_3.pkl'.format(path, dataset),'rb') as f:
            # with open('{}/{}/data_static_2012-01-01_2013-01-01_tt85_ww10.pkl'.format(path, dataset),'rb') as f:
                data_dict = pickle.load(f)
            # times = torch.from_numpy(times)
            y_data = data_dict['y'] # tensor 
            g_data = data_dict['graphs_list'] # dgl
            print(y_data.shape,'y_data',y_data)
            y_data = y_data.sum(-1)
            y_data = torch.where(y_data > 0,1.,0.)
            self.len = len(y_data)
            self.y_data = y_data
            self.g_data = g_data
            print(len(self.g_data),'self.g_data', 'self.y_data',self.y_data.shape)
            # if torch.cuda.is_available():
            #       true_prob_s = true_prob_s.cuda()
            #       true_prob_r = true_prob_r.cuda()
            #       true_prob_o = true_prob_o.cuda()
            #       times = times.cuda()

            # self.times = times
            # self.true_prob_s = true_prob_s
            # self.true_prob_r = true_prob_r
            # self.true_prob_o = true_prob_o

      def __len__(self):
            return self.len

      def __getitem__(self, index):
            return self.g_data[index], self.y_data[index]



def collate_2(batch):
    g_data = [item[0] for item in batch]
    y_data = [item[1] for item in batch]
    return [g_data, y_data]