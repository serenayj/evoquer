import nltk
import h5py
import string
import numpy as np
import string
import torch
import json
from src.utils import utils,io_utils
def j_sim(l1,l2,w):
   # inters=len(list(set(l1).intersection(l2)))
   # union=(len(l1)+len(l2))-inters
    sc=nltk.translate.bleu_score.sentence_bleu([l2],l1,weights=w)
   # return float(inters)/union

    return sc


def sim(path,path1):
    with open(path,"r") as rf:
        js=json.load(rf)
    fobj=open(path1,)
    correct_gt=json.load(fobj)

    st1={}
    st_sim={}
    translator = str.maketrans("", "", string.punctuation)
    res=[]
    for vid in js.keys():
        vv,qid=vid.split('_')
        st=js[vid]
        pred=st["pred"]
        gt=correct_gt[qid][1:-1]
        print(gt)
        st1[vid]={"pred":utils.tokenize(pred.lower(),translator),"gt":gt}
        st_sim[vid]={"sc":j_sim(st1[vid]["pred"],st1[vid]["gt"],(0.5,0.5,))}
        res.append(j_sim(st1[vid]["pred"],st1[vid]["gt"],(0.5,0.5,)))
    print(res)
    print(np.mean(res))

sim("nocontra_gt.json","gt.json")


