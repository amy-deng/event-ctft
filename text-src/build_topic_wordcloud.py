from typing import Counter
import pandas as pd
import numpy as np
import glob, csv, pickle, os, sys
from gensim import corpora, models, similarities
from gensim.models.ldamulticore import LdaMulticore,LdaModel
from gensim.test.utils import common_texts, common_corpus, common_dictionary
from gensim.corpora.dictionary import Dictionary

'''
python build_topic_wordcloud.py THA_2012 THA_2012_50 50
python build_topic_wordcloud.py AFG_2012 AFG_2012_60 60
python build_topic_wordcloud.py EGY_2012 EGY_2012_60 60
python build_topic_wordcloud.py RUS_2012 RUS_2012_60 60

'''

try:
    lda_dict_name = sys.argv[1]
    lda_name = sys.argv[2]
    num_topic = int(sys.argv[3])
except:
    print('Usage: lda_dict_name <THA/THA_2012>, lda_name <THA_50> <num_topic 50>')
    exit()
# country = 'THA'
# lda_name = 'THA_50'
loaded_dict = corpora.Dictionary.load('/home/sdeng/data/icews/topic_models/{}.dict'.format(lda_dict_name))
loaded_lda =  models.LdaModel.load('/home/sdeng/data/icews/topic_models/{}.lda'.format(lda_name))
print('topic model and dictionary loaded')
os.makedirs("/home/sdeng/data/icews/topic_models/{}".format(lda_name),exist_ok=True)

# # save all words and docs in a file
# f = open('/home/sdeng/data/icews/topic_models/{}/top_30_topic_words.csv'.format(lda_name),'a')
# wrt = csv.writer(f)
# wrt.writerow(["topic-id","sorted-words"])#, "event-type", 'rank', "topic-id","effect","z-score","p-value","end-date"])
# for i in range(num_topic):
#     l = [i]
#     for t in loaded_lda.get_topic_terms(i,30):
#         l.append(loaded_dict[int(t[0])])
#     wrt.writerow(l)#, "event-type", 'rank', "topic-id","effect","z-score","p-value","end-date"])
# f.close()

# print('csv file is saved')

from wordcloud import WordCloud, STOPWORDS
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors

cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'
cloud = WordCloud(stopwords=STOPWORDS,
                  background_color='white',
                  width=2200,
                  height=1800,
                  max_words=25,
                  colormap='tab10',
                  color_func=lambda *args, **kwargs: cols[i],
                #   color_func=lambda **kwargs: "black",
                #   color_func=lambda *args, **kwargs: "black",
                  prefer_horizontal=0.8)

topics = loaded_lda.show_topics(num_topics=num_topic,num_words=25,formatted=False)
# 1-19

print(len(topics),topics[0])
# exit()
# topic_indices = [27,33,44,37,10,6,9]
if lda_dict_name[:3] == 'THA':
    topic_indices = [39,6,46,27] # THA
elif lda_dict_name[:3] == 'AFG':
    topic_indices = [27,33,44,37] # AFG
elif lda_dict_name[:3] == 'EGY':
    topic_indices = [17,39,31,52] #
elif lda_dict_name[:3] == 'RUS':
    topic_indices = [18,42,12,42] #  

fig, axes = plt.subplots(2, 2, figsize=(6,6), sharex=True, sharey=True)

for i, ax in enumerate(axes.flatten()):
    # if i >= len(topic_indices):
    #     break
    fig.add_subplot(ax)
    topic_words = dict(topics[topic_indices[i]][1])
#     print((topic_words))
    topic_words_term = {}
    for k in topic_words:
        topic_words_term[loaded_dict[int(k)]] = topic_words[k]
#     print(topic_words_term)
    cloud.generate_from_frequencies(topic_words_term, max_font_size=280)
    plt.gca().imshow(cloud)
    # plt.gca().set_title('Topic ' + str(i), fontdict=dict(size=10))
    plt.gca().axis('off')

    plt.subplots_adjust(wspace=.6, hspace=0)
plt.axis('off')
plt.margins(x=0, y=0)
plt.tight_layout()
path = "/home/sdeng/data/icews/topic_models/{}/wordcloud-causal.pdf".format(lda_name)
print(path)
fig.savefig(path, bbox_inches='tight', dpi=300, transparent=True)
# plt.show()
print('wordcloud of causal topics saved')





# fig, axes = plt.subplots(7, 3, figsize=(10,16), sharex=True, sharey=True)

# for i, ax in enumerate(axes.flatten()):
#     if i >= 20:
#         break
#     fig.add_subplot(ax)
#     topic_words = dict(topics[i][1])
# #     print((topic_words))
#     topic_words_term = {}
#     for k in topic_words:
#         topic_words_term[loaded_dict[int(k)]] = topic_words[k]
# #     print(topic_words_term)
#     cloud.generate_from_frequencies(topic_words_term, max_font_size=300)
#     plt.gca().imshow(cloud)
#     plt.gca().set_title('Topic ' + str(i), fontdict=dict(size=10))
#     plt.gca().axis('off')

#     plt.subplots_adjust(wspace=0, hspace=0)
# plt.axis('off')
# plt.margins(x=0, y=0)
# plt.tight_layout()
# fig.savefig("/home/sdeng/data/icews/topic_models/{}/wordcloud-0-19.pdf".format(lda_name), bbox_inches='tight', dpi=300, transparent=True)
# # plt.show()
# print('wordcloud of topics from 0 to 19 saved')
