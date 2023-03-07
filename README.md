# A Multi-Relational Graph Encoder Network for Fine-Grained Prediction of miRNA-Disease Associations
MicroRNAs (miRNAs) are a class of small, endogenous, non-coding RNAs that play a critical role in the diagnosis and treatment of various diseases. However, biological experiments to uncover the relationship between miRNAs and diseases are expensive and time-consuming. To address this issue, computational methods have been proposed to predict miRNA-disease associations, but more fine-grained approaches are needed.

We propose a multi-relational graph encoder network for fine-grained prediction of miRNA-disease associations (MRFGMDA), which uses practical and current datasets to construct a multi-relational graph encoder network to predict disease-related miRNAs and their specific relationship types (upregulation, downregulation, or dysregulation).
We evaluated MRFGMDA and found that it accurately predicted miRNA-disease associations, which could have far-reaching implications for clinical medical analysis, early diagnosis, prevention, and treatment. Case analyses on three common diseases, including Kaplan-Meier survival analysis, expression difference analysis, and immune infiltration analysis, further demonstrated the effectiveness and feasibility of MRFGMDA to uncover potential disease-related miRNAs.
Overall, our work represents a significant step towards improving the prediction of miRNA-disease associations using a fine-grained approach, which could ultimately lead to more accurate diagnosis and treatment of diseases.
# FlowChat:
The overall workflow of MRFGMDA. Part A, B, and C are carrying out data preprocessing and standardization. Part D is an encoder used to aggregate domain information under different relationships. Part E is a decoder used to calculate the score of candidate triplet information from the feature vector of the node. Part F is a loss function used to evaluate the model's training process and the constraint model.

![FlowChat.png](FlowChat.png)

# Results:
Performance comparison of MRFGMDA with other nine methods with AUC index. We can see from the experimental figure that MRFGMDA has achieved outstanding results compared with the other nine excellent methods.

![Results.png](Results.png)

# Requirements:
* numpy~=1.17.0
* sklearn~=0.0
* scikit-learn~=0.24.2
* tensorflow~=1.14.0
* scipy~=1.5.2
* setuptools~=52.0.0
* theano~=1.0.5

# Train:
* python run_train.py

# Predict:
* python run_predict.py

# Validate:
* python run_valu.py