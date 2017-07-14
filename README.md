# Kaggle-DS-Bowl-17

Kaggle Competition: Data Science Bowl 2017 (Can you improve lung cancer detection?)  
https://www.kaggle.com/c/data-science-bowl-2017  
Author: Li Ding  

---

- **Result**: Bronze Medal, Ranking 107th/1972 (Top 6%)

- **Approach**: Use ResNet-50 with weights pretrained on ImageNet to extract feature of CT scans, which means simply treat them as RGB images.
Then use XGBoost with CV to make prediction.

- **Comment**: Due to limitation of time, didn't try 3D-conv. This is a simple approach using Keras and XGBoost, of which the code can be written in one day and fine-tuned in a week.

---

## How to use:

1. Make sure the raw data are put in the right place.
2. Run feature.py to get .npy features for both stage1 and stage2 data.
3. Run main.py to train XGBoost model and get the submission file.
