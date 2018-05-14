import sys
sys.path.append("/anaconda3/envs/udacity_tensorflow_py3/lib/python3.5/site-packages")
import pandas as pd 
import pickle
import numpy as np

train_file = 'train.csv'
store_file = 'store.csv'
test_file = 'test.csv'
states_file = 'store_states.csv'
df_train=pd.read_csv(train_file)
df_store=pd.read_csv(store_file) 
df_test = pd.read_csv(test_file)
df_states = pd.read_csv(states_file)

df_store["CompetitionDistance"] = df_store["CompetitionDistance"].fillna(0)

df_store = df_store.fillna("NA")

np_train = np.array(df_train)
np_store = np.array(df_store)
np_test = np.array(df_test)
np_states = np.array(df_states)
store_train = []
store_test = []
states_train = []
states_test = []

#matching the store info to the train/test info
for index, value in enumerate(np_train):
    store_train.append(np_store[value[0]-1,1:])
    states_train.append(np_states[value[0]-1,1])

for index, value in enumerate(np_test):
    store_test.append(np_store[value[1]-1,1:])
    states_test.append(np_states[value[1]-1,1])

states_train = np.reshape(states_train,[-1,1])
states_test = np.reshape(states_test,[-1,1])

#combined_train is the concatenation of store info and train data
combined_train = np.hstack([np_train,store_train,states_train])
combined_test = np.hstack([np_test,store_test,states_test])

#reverse sequence for chronological order
combined_train = combined_train[::-1]
combined_test = combined_test[::-1]

col_train = np.hstack([df_train.columns,df_store.columns[1:],df_states.columns[1]])
col_test = np.hstack([df_test.columns,df_store.columns[1:],df_states.columns[1]])

combined_train = pd.DataFrame(combined_train,columns = col_train)
combined_test = pd.DataFrame(combined_test,columns = col_test)

combined_test["Open"] = combined_test["Open"].fillna(1)
combined_test["Open"] = combined_test["Open"].astype("int")

print(combined_train.shape)
print(combined_test.shape)

with open("combined_data_train.pickle","wb") as f:
	pickle.dump(combined_train,f,-1)
with open("combined_data_test.pickle","wb") as f:
	pickle.dump(combined_test,f,-1)

