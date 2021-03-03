# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 09:59:37 2021

@author: shujie.wang
"""

import os
from tqdm import tqdm

#推荐方法
def single_recommend(sim_item_corr, user_item_dict, user_id, top_k, item_num):  
    '''
    Parameters
    ----------
    sim_item_corr : dict
        商品到商品之间的相似度.
    user_item_dict : TYPE
        用户购买的商品序列.
    top_k : int
        推荐时，用户的每个商品选取多少个商品计算相似度.
    item_num : int
        每个用户推荐多少个商品.

    Returns
    -------
    list
        返回的(商品，相似度)列表.
    '''
    rank = {}  
    interacted_items = user_item_dict[user_id]  
    for i in interacted_items:  
        for j, wij in sorted(sim_item_corr[i].items(), key=lambda d: d[1], reverse=True)[0:top_k]:  
            if j not in interacted_items:  
                rank.setdefault(j, 0)
                rank[j] += wij  
    return sorted(rank.items(), key=lambda d: d[1], reverse=True)[:item_num]  

def recommend(item_sim_list,user_item,last_click):
    recom_item = []
    for i in tqdm(last_click['user_id'].unique()):  
        rank_item = single_recommend(item_sim_list, user_item, i, 500, 100)  
        for j in rank_item:  
            recom_item.append([i, j[0], j[1]])  
    return recom_item