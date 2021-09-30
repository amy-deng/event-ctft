import pandas as pd
import numpy as np
import sys, os, json, time, collections, pickle
# import glob


try:
    # event_path = sys.argv[1] # /home/sdeng/data/icews/detailed_event_json/THA_2010_w14h7_city.json
    out_path = sys.argv[1]
    dataset = sys.argv[2] # THA_topic
    raw_data_name = sys.argv[3] 
    # start_year = int(sys.argv[3])
    # end_year = int(sys.argv[4])
    # window = int(sys.argv[3])
    # horizon = int(sys.argv[4])
    # lda_name = sys.argv[5]
    # ngram_path = sys.argv[6]
    # top_k_ngram = int(sys.argv[7])
except:
    print("usage: <out_path> <dataset `THA_topic`> <raw_data_name `raw_w10h7`> ")
    exit()



# train propensity scoring function

# get pairs and calculate average treatment effect 
# for each treatment ele, find a control, most similar

ate 

draw treatment group, control group max()

t  