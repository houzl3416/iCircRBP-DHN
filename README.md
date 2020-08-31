# iCircRBP-DHN
The tool is developed for circRNA-RBP interaction sites identification using deep hierarchical network
![image](https://github.com/houzl3416/iCircRBP-DHN/blob/iCircRBP-DHN/Architecture.png)
# Requirements
- R >= 3.6.1 (64-bit)

- Python 3.7 (64-bit)

- Keras 2.2.0 in Python

- TensorFlow-GPU 1.14.0 in Python

- Numpy 1.18.0 in Python

- Gensim 3.8.3

- Ubuntu 18.04 (64-bit)
# Getting the raw training and test sets
The 37 training sets of circular RNA (circRNA) and 31 testing sets of linear RNA are provided to evaluate our iCircRBP-DHN and compare with other methods.
# Usage
## Setting path
Please change following paths to suit your system for training process of 37 circRNAs datasets:
>***1)*** set the type of the RNA dataset. e.g. ‘set cirRNAID = WTAP’.  
>***2)*** set the type of the RNA Embeddings. e.g. ‘set EmbeddingType = /home/yangyuning/iCircRBP-DHN/circRNA2Vec/circRNA2Vec_model’.  
>***3)*** set the dataset location. e.g. ‘set seqpos_path = '/home/yangyuning/iCircRBP-DHN/Datasets/circRNA-RBP/' + cirRNAID + '/positive', seqneg_path = '/home/yangyuning/iCircRBP-DHN/Datasets/circRNA-RBP/' + cirRNAID + '/negative'.  
>***4)*** set the storage path of model. e.g. ‘set basic_path = /home/yangyuning/iCircRBP-DHN/37results/'.  
## How to train the iCircRBP-DHN model
You can train the model of 5-fold cross-validation with a very simple way by the command blow:  
*Python iCircRBP-DHN.py* and make sure the RNA embedding flag is set to circRNA_model. The script of if **name == "main"** calls training process which trains several models of each model type for a circRNA and finds the best set of hyperparameters. The main function then trains the models several times (num_final_runs) and saves the best model.
## How to train the Linear-RNA model
You can also test the linear-RNA model of 5-fold cross-validation, and make sure the RNA embedding flag is set to linRNA2Vec_model and the file path is set to *linRNA-RBP*.
## How to predict the probability of unknown circRNA
The *iCircRBP-Predict.py* is proposed to calculate the probability for the circRNA-binding protein of unknown types. Please also change following paths to suit your system:
>***1)*** set the sequence location. e.g. ‘set seqPath = ‘/home/Sequence/’.  
>***2)*** set the type of the RNA Embeddings. e.g. ‘set modelType = /home/yangyuning/iCircRBP-DHN/circRNA2Vec/circRNA2Vec_model’.  
>***3)*** set the type of circRNA model. e.g. ‘set modelPredictType = ‘/home/iCircRBP-DHN/result/WTAP/results/model.h5’.  

The prediction results will be displayed automatically. If you need to save the results, please specify the path yourself. Thank you and enjoy the tool! If you have any suggestions or questions, please email me at *yangyn533@nenu.edu.cn*.
