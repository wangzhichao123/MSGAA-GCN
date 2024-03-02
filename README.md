# MSGAA-GCN
## Usage
### 1、Download Google pre-trained PVT-V2 models
### 2、Data preparation:
- **Synapse Multi-organ dataset: **Sign up in the official Synapse website and download the dataset. Then split the 'RawData' folder into 'TrainSet' (18 scans) and 'TestSet' (12 scans) following the TransUNet's lists and put in the './data/synapse/Abdomen/RawData/' folder. Finally, preprocess using python ./utils/preprocess_synapse_data.py or download the preprocessed data and save in the './data/synapse/' folder. Note: If you use the preprocessed data from TransUNet, please make necessary changes (i.e., remove the code segment (line# 88-94) to convert groundtruth labels from 14 to 9 classes) in the utils/dataset_synapse.py.

- **ACDC dataset: **Download the preprocessed ACDC dataset from Google Drive of MT-UNet and move into './data/ACDC/' folder.
