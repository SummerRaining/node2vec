import pandas as pd
import random,os
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from tqdm import tqdm
import numpy as np
import pickle as pkl
import time, math
from node2vec import deepwalk,NodeWalk,train_word2vec
from recommend import recommend

def get_data():
    now_phase = 1  
    train_path = 'kdd_data/underexpose_train'  
    test_path = 'kdd_data/underexpose_test'  
  
    whole_click = pd.DataFrame()
    last_click = pd.DataFrame()
    for c in range(now_phase + 1):  
        print('phase:', c)  
        click_train = pd.read_csv(train_path + '/underexpose_train_click-{}.csv'.format(c), header=None,  names=['user_id', 'item_id', 'time'])  
        click_test = pd.read_csv(test_path + '/underexpose_test_click-{}.csv'.format(c), header=None,  names=['user_id', 'item_id', 'time'])  
  
        #测试集上最后一次的点击分割出来。
        click_test = click_test.sort_values(['user_id','time'],ascending = True).reset_index(drop = True)
        _rn = click_test.groupby(['user_id'])['time'].rank(ascending = False,method='min').reset_index()
        click_test['rn'] = _rn['time']
        last_click = last_click.append(click_test[click_test['rn'] == 1])
        click_test = click_test[click_test['rn'] != 1].drop(['rn'],axis = 1)
        
        whole_click = whole_click.append(click_train)
        whole_click = whole_click.append(click_test)
    return whole_click,last_click

def generate_nodegraph(all_click):
    #用户和商品之间的购买序列,转换成item到item之间的关系图.
    user_item = all_click.groupby(['user_id'])['item_id'].agg(set).reset_index()
    user_item = dict(zip(user_item['user_id'],user_item['item_id']))
    
    item_user = all_click.groupby(['item_id'])['user_id'].agg(set).reset_index()
    item_user = dict(zip(item_user['item_id'],item_user['user_id']))
    node2neibor = {}  
    
    #得到两个商品的相关度，即同时购买这两个商品的用户个数。
    for user_id,item_list in tqdm(user_item.items()): 
        for item1 in item_list:
            node2neibor.setdefault(item1,{})
            for item2 in item_list:
                if item1 != item2:
                    node2neibor[item1].setdefault(item2,0)
                    node2neibor[item1][item2] += 1
    return node2neibor,user_item

def recall_accurracy(model,user_item,last_click):
    #计算 node2vec之间的相似度。
    item_sim_list = {}
    for word in tqdm(model.wv.vocab):
        topn = model.wv.most_similar(positive=[word], topn=500)
        item_sim_list[int(word)] = {int(i):j for i,j in topn}    
        
    recom_item = recommend(item_sim_list, user_item,last_click) #召回
    recom_df = pd.DataFrame(recom_item, columns=['user_id', 'item_id', 'sim']) #计算召回率。  
    recom_df = recom_df.groupby(['user_id'])['item_id'].agg(set).reset_index()
    recom_df = recom_df.merge(last_click[['user_id','item_id']],on = 'user_id')
    recom_df['is_recall'] = recom_df.apply(lambda x:int(x['item_id_y'] in x['item_id_x']),axis = 1)
    print('召回率为{:.4f}'.format(recom_df['is_recall'].mean()))

#使用0-2阶段的数据，来测试。
if __name__ == '__main__':
# =============================================================================
#     deepwalk实例
# =============================================================================
    all_click,last_click = get_data()  #生成数据集
    node2neibor,user_item = generate_nodegraph(all_click) 
    node2neibor = {x:set(node2neibor[x].keys()) for x in node2neibor} #生成商品之间的图
    corpus = deepwalk(node2neibor,n_sample = 40,length = 20) #随机游走
    model = train_word2vec(sentences = corpus,model = 'deepwalk',size = 64,epochs = 3) #生成模型
    recall_accurracy(model,user_item,last_click) #模型用于召回，得到的召回率。
    
# =============================================================================
#     node2vec实例
# =============================================================================
    all_click,last_click = get_data()  #生成数据集
    node2neibor,user_item = generate_nodegraph(all_click) 
    corpus = NodeWalk(node2node = node2neibor,walk_length = 10,walk_num = 40,p=1,q=0.25) #随机游走
    model = train_word2vec(sentences = corpus,model = 'node2vec',size = 32,epochs = 3) #生成模型
    recall_accurracy(model,user_item,last_click)