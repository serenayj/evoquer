# Jason - ActivityNet Folder
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

### 2.1. Current Best Results 
Learning Rate | Batch Size | Update Every | # of Epochs | # of Frames | R1-0.3 | R1-0.5 | R1-0.7 | mIoU
 --- | --- | --- |--- |--- |--- |--- |--- |--- 
0.00004 | 64 | 150 | 500 | 32 | 0.5921 | 0.4202 | 0.2391 | 0.4161

### 2.2. Link to Full Results

https://docs.google.com/spreadsheets/d/1zBbhXyUTm9wjVTwXSBuqmy3PeLn4fiVrhl__73-cH5g/edit?usp=sharing

## 3. Pck-File Results 

https://drive.google.com/drive/folders/1P5x5FuL_C2arpJ5Rkn_VMoM_YeNp5W9D?usp=sharing
