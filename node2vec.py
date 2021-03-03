# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 09:19:12 2021

@author: shujie.wang
"""
import random,os
from tqdm import tqdm
import numpy as np
from gensim.models import Word2Vec

#对deepwalk和node2vec进行训练，对应的词向量。
def train_word2vec(sentences = [],model = 'deepwalk',size = 16,epochs = 3):
    #输入sentence，返回每个item之间的相似度。
    model_path = model+'.model'
    if os.path.exists(model_path):
        model = Word2Vec.load(model_path)
    else:
        model = Word2Vec(size = size, workers=4,hs=1,sg = 1) #0代表cbow，1代表skip-gram，cbow速度快精度低。
        model.build_vocab(sentences)
        model.train(sentences,epochs=epochs,total_examples=model.corpus_count)
        model.save(model_path)     
    return model

#使用deepwalk方法，进行随机游走。
def deepwalk(node2neibor,n_sample = 80, length = 10):
    #deepwalk,对每个节点遍历，随机采样一个邻居，然后邻居在往后采样。一直到序列长度为k。 
    corpus = []
    print('start deep walking!')
    for start_node in tqdm(node2neibor):
        for i in range(n_sample):  #每个节点开始采样10次。
            walk = [start_node]
            while len(walk)<length:
                cur = walk[-1]
                if cur in node2neibor:
                    cur = random.choice(list(node2neibor[cur]))
                    walk.append(cur)
                else:
                    break
            corpus.append(walk)
    corpus = [[str(i) for i in x] for x in corpus]
    return corpus

# 定义node2vec的游走方式。分成node2vec_next_node,startNodeWalk,NodeWalk三个函数。
# 分别定义生成下一个节点，生成一条语句，生成所有的数据集。
def NodeWalk(node2node,walk_length,walk_num,p,q):
    #生成所有的游走序列
    corpus = []
    print('node2vec start random walking!')
    for start_node in tqdm(node2node):
        for i in range(walk_num):
            corpus.append(StartNodeWalk(start_node=start_node,
                walk_length=walk_length,node2node = node2node,p = p,q = q))
    corpus = [[str(i) for i in x] for x in corpus]
    return corpus

def StartNodeWalk(start_node,walk_length,node2node,p = 1,q = 0.25):
    #node2vec的方法进行随机游走。给定起点，生成一条游走序列。
    walk = [start_node]
    while len(walk)<walk_length:
        cur = walk[-1]
        if cur not in node2node:
            break
        if len(walk)==1:
            next_nodes = list(node2node[cur].keys())
            next_probas = [node2node[cur][x] for x in next_nodes]
            next_probas = np.array(next_probas)/sum(next_probas)
            walk.append(np.random.choice(a = next_nodes,p = next_probas))
        else:
            walk.append(node2vec_next_node(a=walk[-2],b=cur,node2node=node2node,p=p,q=q))
    return walk    


def node2vec_next_node(a,b,node2node,p,q):
    #给定a,b点，a是上一跳，b是当前节点。计算下一跳概率，并采样。
    next_nodes = list(node2node[b].keys())
    next_probas = []
    for c in next_nodes:
        weight = node2node[b][c]
        if c == a: #返回上一跳
            next_probas.append(weight/p)
        elif c in node2node[a]: #和a相连
            next_probas.append(weight)
        else: #和a相连
            next_probas.append(weight/q)
    next_probas = np.array(next_probas)/sum(next_probas)
    return np.random.choice(a = next_nodes,p = next_probas)


