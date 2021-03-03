# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 16:19:23 2021

@author: shujie.wang
"""

import pandas as pd  
from tqdm import tqdm  
from collections import defaultdict  
import math,os
import pickle as pkl

from main import generate_nodegraph,get_data
from node2vec import NodeWalk,deepwalk,train_word2vec
from recommend import recommend
  
def get_sim_item(df, user_col, item_col, use_iif=False):  
    user_item_ = df.groupby(user_col)[item_col].agg(set).reset_index()  
    user_item_dict = dict(zip(user_item_[user_col], user_item_[item_col]))  
  
    sim_item = {}  
    item_cnt = defaultdict(int)  
    for user, items in tqdm(user_item_dict.items()):  
        for i in items:  
            item_cnt[i] += 1  
            sim_item.setdefault(i, {})  
            for relate_item in items:  
                if i == relate_item:  
                    continue  
                sim_item[i].setdefault(relate_item, 0)  
                if not use_iif:  
                    sim_item[i][relate_item] += 1  
                else: 
                    sim_item[i][relate_item] += 1 / math.log(1 + len(items))  
    sim_item_corr = sim_item.copy()  
    for i, related_items in tqdm(sim_item.items()):  
        for j, cij in related_items.items():  
            sim_item_corr[i][j] = cij/math.sqrt(item_cnt[i]*item_cnt[j])  
            
    return sim_item_corr, user_item_dict  

def get_bipartile_sim_item(df, user_col, item_col):  
    user_item_ = df.groupby(user_col)[item_col].agg(set).reset_index()  
    user_item_dict = dict(zip(user_item_[user_col], user_item_[item_col])) #用户id到item id的字典、
    
    item_user_ = df.groupby(item_col)[user_col].agg(set).reset_index()  
    item_user_dict = dict(zip(item_user_[item_col], item_user_[user_col])) #商品id到用户id的字典
    
    sim_item = {} #商品的相似度字典。
    for item, users in tqdm(item_user_dict.items()):
        sim_item.setdefault(item, {}) 
        for u in users:
            tmp_len = len(user_item_dict[u]) #计算该用户的度。
        
            for relate_item in user_item_dict[u]:
                sim_item[item].setdefault(relate_item, 0) #默认目标商品，和相邻商品的权重为0。
                sim_item[item][relate_item] += 1/ (math.log(len(users)+1) * math.log(tmp_len+1)) #目标商品和相邻商品的权重为，连接用户的度*目标商品的度取log的倒数。
                #和公式一致，将公式中的度替换为，度+1取log，增加了流行商品的重要性。
    return sim_item, user_item_dict  
  
def get_sim_item_deep_walk(all_click):
    '''
    输入点击数据，返回node和node之间的相似度

    Parameters
    ----------
    all_click : dataframe
        用于训练的点击数据
    '''
    node2neibor,user_item = generate_nodegraph(all_click) 
    node2neibor = {k:set(v.keys()) for k,v in node2neibor.items()}
    if os.path.exists('DeepWalkCorpus.pkl'):
        corpus = pkl.load(open('DeepWalkCorpus.pkl','rb'))
    else:
        corpus = deepwalk(node2neibor,n_sample = 40,length = 20) #最大深度为10.
        pkl.dump(corpus,open('DeepWalkCorpus.pkl','wb'))
    model = train_word2vec(sentences = corpus,model = 'deepwalk',size = 32,epochs = 3)
    #计算 node之间的相似度。
    item_sim_list = {}
    for word in tqdm(model.wv.vocab):
        topn = model.wv.most_similar(positive=[word], topn=500)
        item_sim_list[int(word)] = {int(i):j for i,j in topn}
    return item_sim_list,user_item
 

def get_sim_item_node2vec(all_click):
    node2neibor,user_item = generate_nodegraph(all_click) 
    if os.path.exists('NodeCorpus.pkl'):
        corpus = pkl.load(open('NodeCorpus.pkl','rb'))
    else:
        corpus = NodeWalk(node2node = node2neibor,walk_length=10,walk_num=40,p=1,q=0.25)
        pkl.dump(corpus,open('NodeCorpus.pkl','wb'))
    model = train_word2vec(sentences = corpus,model = 'node2vec',size = 32,epochs = 3) #生成模型
    #计算 node2vec之间的相似度。
    item_sim_list = {}
    for word in tqdm(model.wv.vocab):
        topn = model.wv.most_similar(positive=[word], topn=500)
        item_sim_list[int(word)] = {int(i):j for i,j in topn}    
    return item_sim_list,user_item

if __name__ == "__main__":  
    all_click,last_click = get_data()
    
    item_sim_list0, user_item = get_sim_item_node2vec(all_click)
    item_sim_list1, user_item = get_sim_item(all_click, 'user_id', 'item_id', use_iif=True)  
    item_sim_list2, user_item = get_bipartile_sim_item(all_click, 'user_id', 'item_id')  
    item_sim_list3, user_item = get_sim_item_deep_walk(all_click)

    recom_item0 = recommend(item_sim_list0, user_item,last_click)
    recom_item1 = recommend(item_sim_list1, user_item,last_click)
    recom_item2 = recommend(item_sim_list2, user_item,last_click)    
    recom_item3 = recommend(item_sim_list3, user_item,last_click)
    recom_item = recom_item1+recom_item2+recom_item3+recom_item0
    
    recom_df = pd.DataFrame(recom_item, columns=['user_id', 'item_id', 'sim'])  
    recom_df = recom_df.groupby(['user_id'])['item_id'].agg(set).reset_index()
    recom_df = recom_df.merge(last_click[['user_id','item_id']],on = 'user_id')
    recom_df['is_recall'] = recom_df.apply(lambda x:int(x['item_id_y'] in x['item_id_x']),axis = 1)
    print(recom_df['is_recall'].mean())
    
    for recom_item in [recom_item1,recom_item2,recom_item3,recom_item0]:
        recom_df = pd.DataFrame(recom_item, columns=['user_id', 'item_id', 'sim'])  
        recom_df = recom_df.groupby(['user_id'])['item_id'].agg(set).reset_index()
        recom_df = recom_df.merge(last_click[['user_id','item_id']],on = 'user_id')
        recom_df['is_recall'] = recom_df.apply(lambda x:int(x['item_id_y'] in x['item_id_x']),axis = 1)
        print(recom_df['is_recall'].mean())
        