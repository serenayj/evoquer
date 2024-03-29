from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import h5py
import string
import numpy as np
import string
import torch
import json
import time

import torch
import torch.utils.data as data

from collections import defaultdict, OrderedDict
from tqdm import tqdm
from abc import abstractmethod
from src.dataset.abstract_dataset import AbstractDataset
from src.utils import utils,io_utils

def create_loaders(split, loader_configs, num_workers):
        dsets, L = {}, {}
        for di,dt in enumerate(split):
                shuffle = True if dt == "train" else False
                drop_last = True if dt == "train" else False
                dsets[dt] = ActivityNetCaptionsDataset(loader_configs[di])
                L[dt] = data.DataLoader(
                        dsets[dt],
                        batch_size = loader_configs[di]["batch_size"],
                        num_workers = num_workers,
                        shuffle = shuffle, # shuffle
                        collate_fn = dsets[dt].collate_fn,
                        drop_last= drop_last #drop_last
                )
        return dsets, L

class ActivityNetCaptionsDataset(AbstractDataset):
        def __init__(self, config):
                super(self.__class__, self).__init__(config)
                # get options
                self.S = config.get("num_segment", 128)
                self.split = config.get("split", "train")
                self.data_dir = config.get("data_dir", "")
                self.feature_type = config.get("feature_type", "C3D")
                self.in_memory = config.get("in_memory", False)
                self.feat_hdf5 = "/home/lzl5409/vpmt-master/src/scripts/data/LGI/ActivityNet/feats/sub_activitynet_v1-3.c3d.hdf5"
    #Users//Desktop/LGI4temporalgrounding-master/data/ActivityNet/feats/sub_activitynet_v1-3.c3d.hdf5"

                # get paths for proposals and captions
                paths = self._get_data_path(config)
                print("GETTING PATHS ", paths)
                # create labels (or load existing one)
                ann_path = config.get("annotation_path",
                                "/home/lzl5409/vpmt-master/src/scripts/data/LGI/ActivityNet/captions/annotations/train.json")
                self.anns, self.qids, self.vids  = self._load_annotation(ann_path)
                #if not self._exist_data(paths):
                self.generate_labels(config)

                # load features if use in_memory
                if self.in_memory:
                        self.feats = {}
                        h = io_utils.load_hdf5(self.feat_hdf5, verbose=False)
                        for k in tqdm(self.vids, desc="In-Memory: vid_feat"):
                                self.feats[k] = h[k]["c3d_features"][:]

                        self.s_pos, self.e_pos, self.att_mask = {}, {}, {}
                        grd_info = io_utils.load_hdf5(self.paths["grounding_info"])
                        for k in tqdm(self.qids, desc="In-Memory: grounding"):
                                self.s_pos[k] = grd_info["start_pos/"+k][()]
                                self.e_pos[k] = grd_info["end_pos/"+k][()]
                                self.att_mask[k] = grd_info["att_mask/"+k][()]

                        self.query_labels = {}
                        query_labels = h5py.File(self.paths["query_labels"], "r")
                        for k in tqdm(self.qids, desc="In-Memory: query"):
                                self.query_labels[k]= query_labels[k][:]

                # load and prepare json files
                query_info = io_utils.load_json(self.paths["query_info"])
                self.wtoi = query_info["wtoi"]
                self.itow = query_info["itow"]
                self.query_lengths = query_info["query_lengths"]

                self.batch_size = config.get("batch_size", 100)
                self.num_instances = len(self.qids)

        def __getitem__(self, idx):
                # get query id and corresponding video id
                qid = str(self.qids[idx])
                vid = self.anns[qid]["video_id"]
                timestamp = self.anns[qid]["timestamps"]
                duration = self.anns[qid]["duration"]
                qry=self.anns[qid]["query"]
                # Descriptions
                #d_label_lsts = self.descriploader.encoded_description(vid, self.wtoi)
                #description_length, description_labels = d_label_lsts['descrip_lengths'], d_label_lsts['descrip_labels']

                # get query labels
                if self.in_memory:
                        q_label = self.query_labels[qid]
                else:
                        query_labels = h5py.File(self.paths["query_labels"], "r")
                        q_label = query_labels[qid][:]
                q_leng = self.query_lengths[qid]

                # get grounding label
                if self.in_memory:
                        start_pos = self.s_pos[qid]
                        end_pos = self.e_pos[qid]
                else:
                        grd_info = h5py.File(self.paths["grounding_info"],"r")
                        start_pos = grd_info["start_pos/"+qid][()]
                        end_pos = grd_info["end_pos/"+qid][()]

                # get video features
                if self.in_memory:
                        vid_feat_all = self.feats[vid]
                else:
                        vid_feat_all = io_utils.load_hdf5(self.feat_hdf5, verbose=False)[vid]["c3d_features"]
                vid_feat, nfeats, start_index, end_index = self.get_fixed_length_feat(
                                vid_feat_all, self.S, start_pos, end_pos)

                # get video masks
                vid_mask = np.zeros((self.S, 1))
                vid_mask[:nfeats] = 1

                # get attention mask
                if self.in_memory:
                        att_mask = self.att_mask[qid]
                else:
                        att_mask = grd_info["att_mask/"+qid][:]
                instance = {
                        "vids": vid,
                        "qids": qid,
                        "timestamps": timestamp, # GT location [s, e] (seconds)
                        "duration": duration, # video span (seconds)
                        #"description_length": description_length,
                        #"description_labels": description_labels,
                        "query_lengths": q_leng,
                        # "query":qry,
                        "query_labels": torch.LongTensor(q_label).unsqueeze(0), 
                        "query_masks": (torch.FloatTensor(q_label)>0).unsqueeze(0), # [1,L_q_max]
                        "grounding_start_pos": torch.FloatTensor([start_pos]), # [1]; normalized
                        "grounding_end_pos": torch.FloatTensor([end_pos]),     # [1]; normalized
                        "grounding_att_masks": torch.FloatTensor(att_mask),  # [L_v]
                        "nfeats": torch.FloatTensor([nfeats]),
                        "video_feats": torch.FloatTensor(vid_feat), # [L_v,D_v]
                        "video_masks": torch.ByteTensor(vid_mask), # [L_v,D-v]
                }
                
                return instance
        def collate_fn(self,data):
            seq_items=["video_feats","video_masks","grounding_att_masks"]
            tensor_items=["query_labels","query_masks","nfeats","grounding_start_pos","grounding_end_pos",]
            batch={k:[d[k] for d in data] for k in data[0].keys()}
            if len(data)==1:
                for k,v in batch.items():
                    if k in tensor_items:
                        batch[k]=torch.cat(batch[k],0)
                    elif k in seq_items:
                        batch[k]=torch.nn.utils.rnn.pad_sequence(batch[k], batch_first=True)
                    else:
                        batch[k]=batch[k][0]
            else:
                for k in tensor_items:
                    batch[k]=torch.cat(batch[k],0)
                for k in seq_items:
                    batch[k]=torch.nn.utils.rnn.pad_sequence(batch[k], batch_first=True)
            return batch

        def _preprocessing(self,path,new_anns,vids,qstart):
                with open(path,"r") as rf:
                        anns=json.load(rf)
                qid=qstart
                translator = str.maketrans("", "", string.punctuation)
                for vid in anns.keys():
                        ann=anns[vid]
                        duration=ann["duration"]
                        descrip=" ".join(ann["sentences"])
                        for ts,q in zip(ann["timestamps"], ann["sentences"]):
                            file1 = open("simple_query.txt","a")
                            file1.write(str(qid)+"##"+q+"\n")
        
                            #with open("simple_query.json","w") as f:
                            #    json.dump({qid:q},f)
                            new_anns[str(qid)]={"timestamps":ts,"query":q,"descriptions":descrip,"tokens":utils.tokenize(q.lower(),translator),"duration":duration,"video_id":vid}
                            qid += 1

                        vids.extend(list(anns.keys()))
                return new_anns, qid,  list(set(vids))

        def _build_vocab(self,anns, frequency_threshold=1):
                # count frequencies of words from captions
                frequency = defaultdict(lambda: 0)
                for qid,ann in tqdm(anns.items(), desc="Count frequency"):
                                for w in ann["tokens"]:
                                                frequency[w] += 1
                                # thresholding vocabulary
                cw = sorted([(cnt,w) for w,cnt in frequency.items()], reverse=True)
                words = sorted([w for w,cnt in frequency.items() if cnt >= frequency_threshold])
                before = len(words)

                # construct vocabulary
                wtoi = OrderedDict() # mapping word to index
                wtoi["<PAD>"], wtoi["<UNK>"] = 0, 1 # PAD, UNKNOWN tokens
                wtoi["<S>"], wtoi["<E>"]     = 2, 3 # START, END tokens
                for wi,w in enumerate(set(words)):
                                wtoi[w] = wi + 4 # words start in 4
                return wtoi

        def _encode_query(self,anns, wtoi, max_length=25):
                labels, lengths = {}, {}
                for qid,ann in tqdm(anns.items(), desc="Encoding query"):
                        tokens = ann["tokens"]
                        # obtain query labels and their lengths
                        lengths[qid] = min(len(tokens), max_length)
                        labels[qid] = np.zeros((max_length), dtype=np.int)
                        # words -> labels
                        for wi,w in enumerate(tokens):
                                if wi == max_length: break
                                labels[qid][wi] = wtoi.get(w, 1)   # 1: <UNK> token

                        encoded = {
                                        "query_lengths": lengths,
                                        "query_labels": labels,
                        }
                return encoded

        def _get_data_path(self, config):

                split = config.get("split", "train")
                L = config.get("max_length", 20)
                F = config.get("frequency_threshold", 1)
                S = config.get("num_segment", 128)
                FT = config.get("feature_type", "C3D")

                root_dir = config.get("data_dir")

                query_path=os.path.join(root_dir,"simple_query",split+"_simple_query.txt")

                grounding_info_path = os.path.join(root_dir,
                        "grounding_info", split+"_grounding_info.hdf5")
                #Rename
                query_info_path = os.path.join(root_dir,
                        "query_info", split+"_query_info.json")
                query_label_path = os.path.join(root_dir, "query_labels", split+"_query_label.hdf5")

                #query_label_path = os.path.join(root_dir,
                #               "query_info", "{}_label_F{}_L{}_{}.hdf5".format(split, F, L, FT))
                #caption_label_path = os.path.join(root_dir,
                #               "query_info", "{}_caption_label_F{}_L{}_{}.hdf5".format(split, F, L, FT))

                io_utils.check_and_create_dir(os.path.join(root_dir, "grounding_info"))
                io_utils.check_and_create_dir(os.path.join(root_dir, "query_labels"))
                io_utils.check_and_create_dir(os.path.join(root_dir, "query_info"))
                io_utils.check_and_create_dir(os.path.join(root_dir, "simple_query"))

                self.paths = {
                        "grounding_info": grounding_info_path,
                        "query_labels": query_label_path,
                        "query_info": query_info_path,
                        "simple_query": query_path
                }
                return self.paths

        def _load_annotation(self, ann_path):
                """ Load annotations
                Args:
                        ann_paths: path for annotations; list or string
                Returns:
                        new_anns: loaded and preprocessed annotations
                """
                qid = 0
                new_anns = {}
                vids = []
                if isinstance(ann_path, list):
                        # for validation annotation
                        for ap in ann_path:
                           # anno = io_utils.load_json(ap)
                            new_anns, qid, vids = self._preprocessing(ap, new_anns, vids,qid)

                else:
                        # for train annotation
                       # anno = io_utils.load_json(ann_path)
                        new_anns, qid, vids = self._preprocessing(ann_path, new_anns, vids, qid)

                return new_anns, list(new_anns.keys()), vids



        def generate_labels(self, config):
                """ Generate and save labels for temporal language grouding
                        1)query_info (.json) with
                        - wtoi: word to index dictionary (vocabulary)
                        - itow: index to word dictionary (vocabulary)
                        - query_lengths: lengths for queries
                        3)grounding_labels (.h5): qid -> label
                """


                """ Query information """
                if not os.path.exists(self.paths["query_labels"]):
                        # build vocabulary from training data
                        train_ann_path = "/home/lzl5409/vpmt-master/src/scripts/data/LGI/ActivityNet/captions/annotations/train.json"
                        train_anns, _, _= self._load_annotation(train_ann_path)
                        wtoi = self._build_vocab(train_anns)
                        itow = {v:k for k,v in wtoi.items()}


                #########################################################################################
                        # encode query and save labels (+lenghts)
                        L = config.get("max_length", 25)
                        encoded = self._encode_query(self.anns, wtoi, L)
                        query_labels = io_utils.open_hdf5(self.paths["query_labels"],"w")
                        for qid in tqdm(encoded["query_lengths"].keys(), desc="Saving query"):
                            _ = query_labels.create_dataset(str(qid), data=encoded["query_labels"][qid])
                #  query_labels.close()

                        # save vocabulary and query length
                        query_info = {
                        "wtoi": wtoi,
                        "itow": itow,
                        "query_lengths": encoded["query_lengths"],
                        }
                        io_utils.write_json(self.paths["query_info"], query_info)
                #########################################################################################

                """ Grounding information """
                if not os.path.exists(self.paths["grounding_info"]):
                        #if self.feature_type == "C3D":
                        features = io_utils.load_hdf5(self.feat_hdf5)

                        grd_dataset = io_utils.open_hdf5(self.paths["grounding_info"],"w")
                        start_pos = grd_dataset.create_group("start_pos")
                        end_pos = grd_dataset.create_group("end_pos")
                        att_masks = grd_dataset.create_group("att_mask")
                        for qid,ann in tqdm(self.anns.items(), desc="Gen. Grd. Labels"):
                            # get starting/ending positions
                            ts = ann["timestamps"]
                            vid_d = ann["duration"]
                            start = ts[0] / vid_d
                            end = ts[1] / vid_d
                            # get attention calibration mask
                            vid = ann["video_id"]
                            nfeats = features[vid]["c3d_features"][:].shape[0]
                            nfeats = min(nfeats, self.S)
                            fs = utils.timestamp_to_featstamp(ts, nfeats, vid_d)
                            att_mask = np.zeros(self.S)
                            att_mask[fs[0]:fs[1]+1]=1

                            _ = start_pos.create_dataset(qid,data=start,dtype="float")
                            _ = end_pos.create_dataset(qid,data=end,dtype="float")
                            _ = att_masks.create_dataset(qid,data=att_mask,dtype="float")
                        grd_dataset.close()

                if not os.path.exists(self.paths["simple_query"]):

                        for qid,ann in tqdm(self.anns.items(), desc="Gen. Grd. Labels"):
                            file1 = open(self.paths["simple_query"],"a")
                            file1.write(str(qid)+"##"+ann["query"]+"\n")

                    
# for debugging
def get_loader():
        conf = {
                "train_loader": {
                        "dataset": "activitynet",
                        "split": "train",
                        "batch_size": 100,
                        "data_dir": "/home/lzl5409/vpmt-master/data/anet",
                        "annotation_path": "/home/lzl5409/vpmt-master/src/scripts/data/LGI/ActivityNet/captions/annotations/train.json",
                        "video_feature_path":"/home/lzl5409/vpmt-master/src/scripts/data/LGI/ActivityNet/feats/sub_activitynet_v1-3.c3d.hdf5",
                        "max_length": 25,
                        "word_frequency_threshold": 1,
                        "num_segment": 128,
                        "feature_type": "C3D",
                },
                "test_loader": {
                        "dataset": "activitynet",
                        "split": "val",
                        "batch_size": 100,
                        "data_dir": "/home/lzl5409/vpmt-master/data/anet",
                        "annotation_path":
                                ["/home/lzl5409/vpmt-master/src/scripts/data/LGI/ActivityNet/captions/annotations/val_1.json",
                                "/home/lzl5409/vpmt-master/src/scripts/data/LGI/ActivityNet/captions/annotations/val_2.json"],
                        "video_feature_path": "/home/lzl5409/vpmt-master/src/scripts/data/LGI/ActivityNet/feats/sub_activitynet_v1-3.c3d.hdf5",
                        "max_length": 25,
                        "word_frequency_threshold": 1,
                        "num_segment": 128,
                        "feature_type": "C3D",
                }
        }
        print(json.dumps(conf, indent=4))
        dsets, L = create_loaders(["train","test"],
                                                          [conf["train_loader"], conf["test_loader"]],
                                                          num_workers=4)
        return dsets, L

if __name__ == "__main__":
        i = 1
        dset, l = get_loader()
        D=dset["test"]
        vis_data=D.get_samples(int(4))

       # bt = time.time()
       # st = time.time()
        # for batch in l["train"]:
        #       print("=====> {}th training batch ({:.5f}s)".format(i, time.time() - st))
        #       i += 1
        #       st = time.time()
       # i = 1
       # for batch in l["test"]:
        #        print("=====> {}th test batch ({:.5f}s)".format(i, time.time() - st))
        #        i += 1
        #        st = time.time()
       # print("Total elapsed time ({:.5f}s)".format(time.time() - bt))
