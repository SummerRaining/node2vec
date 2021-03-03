# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 15:36:44 2021

@author: shujie.wang
"""

from data_preprocess import get_data
from tqdm import tqdm
import numpy as np
def build_graph(all_click):
    #构建商品的网络图。
    user_item = all_click.groupby(['user_id'])['item_id'].agg(set).reset_index()
    user_item = dict(zip(user_item['user_id'],user_item['item_id']))
    
    item_user = all_click.groupby(['item_id'])['user_id'].agg(set).reset_index()
    item_user = dict(zip(item_user['item_id'],item_user['user_id']))
    node2neibor = {}  #每个节点，和它相邻的节点。
    
    #修改采样的方法，得到两个商品的相关度。即同时购买这两个商品的用户个数。
    for user_id,item_list in tqdm(user_item.items()): 
        for item1 in item_list:
            node2neibor.setdefault(item1,{})
            for item2 in item_list:
                if item1 != item2:
                    node2neibor[item1].setdefault(item2,0)
                    node2neibor[item1][item2] += 1
    return node2neibor,user_item,item_user

if __name__ == '__main__':
    all_click,last_click = get_data()
    node2neibor,user_item,item_user = build_graph(all_click)
    
    #给定两个节点a，b计算它们的二阶相似度AA_Sim
    node2node = node2neibor
    def AA_sim_twoNode(a,b,node2node,item_user):
        relate_node = node2node[a].keys() & node2node[b].keys()
        sim = 0
        if len(relate_node) != 0:
            for c in relate_node:
                sim += node2node[a][c]*node2node[b][c]/np.log(1+len(item_user[c]))
        return sim
    
    #时间太长20小时以上+。
    AA_sim = {}
    for node1 in tqdm(node2neibor):
        AA_sim.setdefault(node1,{})
        for node2 in node2neibor:
            AA_sim[node1][node2] = AA_sim_twoNode(node1,node2,node2neibor,item_user)
            