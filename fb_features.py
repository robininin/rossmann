import sys
sys.path.append("/anaconda3/envs/udacity_tensorflow_py3/lib/python3.5/site-packages")
import pandas as pd
import numpy as np
import pickle
# This script is used to generate forward_backward info of open, promo, schoolholiday, and hstack with original combined data of
# train and testing

with open("combined_data_train.pickle","rb") as f:
    df_combined_train = pickle.load(f)
    
with open("combined_data_test.pickle","rb") as f:
    df_combined_test = pickle.load(f)

def backward(promo, column, window=4):
    #used on open, promo, schoolholiday for one store. promo is pd series, column is string. "1000" means yesterday was 1
    indexes = promo.index
    num = len(promo)
    result = []
    for i in range(num):
        if i < window:
            result.append("0000")
            continue
        else:
            result.append(str(promo[indexes[i-1]]))
            for j in range(window-1):
                result[i] += str(promo[indexes[i-j-2]])
    result = pd.DataFrame(result,index=promo.index,columns=[column])
    return result

def forward(promo, column, window=4):
    #used on open, promo, schoolholiday for one store. promo is pd series, column is string. "1000" means tmr is 1
    indexes = promo.index
    num = len(promo)
    result = []
    for i in range(num):
        if i+window > num-1:
            result.append("0000")
        else:
            result.append(str(promo[indexes[i+1]]))
            for j in range(window-1):
                result[i] += str(promo[indexes[i+j+2]])
    result = pd.DataFrame(result,index=promo.index,columns=[column])
    return result

def prepare_fb(data):
    #the data should be in chronological order and the index should be in ascending order
    #this function will return the sorted dataframes of fb features
    unique_store = np.unique(data["Store"])
    
    backward_open_total = pd.DataFrame(columns=["backward_open"])
    forward_open_total = pd.DataFrame(columns=["forward_open"])    
    backward_promo_total = pd.DataFrame(columns=["backward_promo"])
    forward_promo_total = pd.DataFrame(columns=["forward_promo"])
    backward_school_total = pd.DataFrame(columns=["backward_school"])
    forward_school_total = pd.DataFrame(columns=["forward_school"])
    
    for store in unique_store:
        store_data = data[data["Store"]==store]
        backward_open = backward(store_data["Open"], "backward_open")
        forward_open = forward(store_data["Open"], "forward_open")
        backward_promo = backward(store_data["Promo"], "backward_promo")
        forward_promo = forward(store_data["Promo"], "forward_promo")
        backward_school = backward(store_data["SchoolHoliday"], "backward_school")
        forward_school = forward(store_data["SchoolHoliday"], "forward_school")
        
        backward_open_total = backward_open_total.append(backward_open)
        forward_open_total = forward_open_total.append(forward_open)
        backward_promo_total = backward_promo_total.append(backward_promo)
        forward_promo_total = forward_promo_total.append(forward_promo)
        backward_school_total = backward_school_total.append(backward_school)
        forward_school_total = forward_school_total.append(forward_school)
        
    backward_open_total = backward_open_total.sort_index()
    forward_open_total = forward_open_total.sort_index()  
    backward_promo_total = backward_promo_total.sort_index()
    forward_promo_total = forward_promo_total.sort_index()
    backward_school_total = backward_school_total.sort_index()
    forward_school_total = forward_school_total.sort_index()
    
    return {"backward_open": backward_open_total, "forward_open": forward_open_total, "backward_promo": backward_promo_total,
           "forward_promo": forward_promo_total, "backward_school": backward_school_total, "forward_school": forward_school_total}

def fb2int(fb):
    d = {'0000' : 0, '0001' : 1, '0010' : 2, '0011' : 3, '0100' : 4, '0101' : 5, '0110' : 6, '0111' : 7,
       '1000' : 8, '1001' : 9, '1010' : 10, '1011' : 11, '1100' : 12, '1101' : 13, '1110' : 14, '1111' : 15}
    result = fb.apply(lambda x:d[x])
    return result

fb_dict_train = prepare_fb(df_combined_train)
fb_dict_test = prepare_fb(df_combined_test)

backward_open_train = np.reshape((fb2int(fb_dict_train["backward_open"]["backward_open"])),(-1,1))
forward_open_train = np.reshape(fb2int(fb_dict_train["forward_open"]["forward_open"]),(-1,1))
backward_promo_train = np.reshape(fb2int(fb_dict_train["backward_promo"]["backward_promo"]),(-1,1))
forward_promo_train = np.reshape(fb2int(fb_dict_train["forward_promo"]["forward_promo"]),(-1,1))
backward_school_train = np.reshape(fb2int(fb_dict_train["backward_school"]["backward_school"]),(-1,1))
forward_school_train = np.reshape(fb2int(fb_dict_train["forward_school"]["forward_school"]),(-1,1))

backward_open_test = np.reshape(fb2int(fb_dict_test["backward_open"]["backward_open"]),(-1,1))
forward_open_test = np.reshape(fb2int(fb_dict_test["forward_open"]["forward_open"]),(-1,1))
backward_promo_test = np.reshape(fb2int(fb_dict_test["backward_promo"]["backward_promo"]),(-1,1))
forward_promo_test = np.reshape(fb2int(fb_dict_test["forward_promo"]["forward_promo"]),(-1,1))
backward_school_test = np.reshape(fb2int(fb_dict_test["backward_school"]["backward_school"]),(-1,1))
forward_school_test = np.reshape(fb2int(fb_dict_test["forward_school"]["forward_school"]),(-1,1))

combined_train = np.array(df_combined_train)
combined_test = np.array(df_combined_test)

train_fb = np.hstack([combined_train, backward_open_train, forward_open_train, backward_promo_train, forward_promo_train, backward_school_train, forward_school_train])
test_fb = np.hstack([combined_test, backward_open_test, forward_open_test, backward_promo_test, forward_promo_test, backward_school_test, forward_school_test])

col_df = ['backward_open','forward_open','backward_promo','forward_promo','backward_school','forward_school']
col_train = np.hstack([df_combined_train.columns, col_df])
col_test = np.hstack([df_combined_test.columns, col_df])

df_train_fb = pd.DataFrame(train_fb, columns=col_train)
df_test_fb = pd.DataFrame(test_fb, columns=col_test)

print(df_train_fb.shape)
print(df_test_fb.shape)

with open("combined_fb_train.pickle","wb") as f:
    pickle.dump(df_train_fb,f,-1)
with open("combined_fb_test.pickle","wb") as f:
    pickle.dump(df_test_fb,f,-1)
