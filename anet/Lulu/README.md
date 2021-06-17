Lulu's ActivityNet Folder
#  - ActivityNet Folder
ActivityNet(anet)

## 1. File Descriptions

`VPMT_eric.py`: EVOQUER Pipeline including models, optimizer, forward functions, & update

`anet.py`: Dataloader for ActivityNet Dataset

`config_anet.yml`: Settings for training the nerual network

`eric_mt.py`: Encoder/Decoder

`pipeline_utils.py`: Simplfied translation functions

`preprocess_query_simpl_trans.py`: Script in order to simplify input queries

`test_translate_eric.py`: File to run in translation module

`trainer_eric.py`: File to run in order to train neural network (LGI)

## 2. Spreadsheet of Results 

`I've ran two experiments for ActivityNet
One with learning rate=0.0004,
         lr_update_every=150,
         batch_size=128,
         epoch=600,
         and frames=32, the results are in 0429.txt and scores_429.log.'
         
'The other with lr= 0.00004,
               batch_size = 128,
               lr_update_every = 150,
               epoch = 1500,
               and frames = 32, we cut the dataset to be the half, and simplify the the vocabulary for the translation to be 4961.
               The results are in 0513.txt and scores_0513.log. pkl file is also generated as 2021-05-13anet_eric_32f_1500_nocontra_Separate_vpmt.pkl.
`
### 2.1. Current Best Results 
Learning Rate | Batch Size | Update Every | # of Epochs | # of Frames | R1-0.3 | R1-0.5 | R1-0.7 | mIoU
 --- | --- | --- |--- |--- |--- |--- |--- |--- 
0.0004 | 64 | 150 | 600 | 32 | 0.5857 | 0.4165 | 0.2393 | 0.4139

### 2.2. Link to Full Results

https://docs.google.com/spreadsheets/d/1zBbhXyUTm9wjVTwXSBuqmy3PeLn4fiVrhl__73-cH5g/edit?usp=sharing

## 3. Pck-File Results 

https://drive.google.com/drive/folders/1P5x5FuL_C2arpJ5Rkn_VMoM_YeNp5W9D?usp=sharing


               





 
