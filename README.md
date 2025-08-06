This work is part of paper **CapTrAD: A Hybrid Model for Alzheimer’s Disease Classification** as post publication in 

## Abstract

Accurate classification of Alzheimer’s disease (AD) stages using brain MRI is essential for timely intervention. This study proposes CapsFormer, a novel hybrid deep learning model that integrates a ResNet backbone, capsule layers for part–whole feature representation, and a Transformer encoder to model long-range dependencies. The model was evaluated on a four-class classification task (healthy control, mild, moderate, and severe AD) using structural MRI data. CapsFormer outperformed previous approaches in terms of precision, recall, F1-score, and AUC across all classes. Visual explanations were obtained using Grad-CAM to enhance interpretability. Statistical validation using McNemar’s test confirmed the significance of the performance improvement. These results suggest that CapsFormer provides a robust and explainable framework for AD stage classification.

## Results

This study introduced CapsFormer, a novel hybrid architecture combining ResNet-based feature extraction, capsule layers for part–whole relationships, and a Transformer encoder for modeling long-range dependencies. The model demonstrated superior performance in multiclass classification of Alzheimer's disease stages from structural MRI, achieving high precision, recall, F1-scores, and AUC across all classes. Visual explanations using Grad-CAM and statistical validation with the McNemar test further reinforced the robustness and interpretability of the proposed approach.

Future research will explore several extensions. First, the model will be evaluated on additional neuroimaging modalities such as PET and rs-fMRI to assess multimodal performance. Second, scalability to larger and more heterogeneous datasets will be investigated to improve generalizability. Finally, interpretability will be enhanced through advanced explainable AI techniques beyond Grad-CAM, such as attention rollouts or concept-based methods, to support clinical applicability better.

