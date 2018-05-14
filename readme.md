Rossmann Sales Prediction

Kaggle competition: https://www.kaggle.com/c/rossmann-store-sales

通过融合10个Entity Embedding+Neural Network在Private Leaderboard上得分0.11098,如果和参加比赛的3303只队伍比较，最后得分排名26

运行所需库：
pickle, keras, pandas, numpy, seaborn, matplotlib, math, datetime, isoweek

如果需要重现模型，请依次运行以下文件：

extract_data.py
fb_features.py （笔记本上约10分钟）
feature_prep.py （笔记本上约2分钟）
run.py

一个模型默认训练20个epochs。20个epochs在本地笔记本电脑Mac Air CPU上训练需要50分钟，在AWS p3.2xlarge实例上训练需要20分钟。预测时所需时间很短。
可以通过调整model_adj_embedding.py中的self.epoch参数调整epoch大小。

默认运行一个模型，并在所有训练数据上训练。可以在run.py文件中改变num_network和testing的值。如果testing选择False，将会从训练集中分割出3%作为模拟测试集，
并且在剩下的97%训练数据中再分割出2%作为验证集。

运行完后，可以在testing.ipynb中进行验证数据可视化，模拟测试集测试，以及生成prediction.csv文件


