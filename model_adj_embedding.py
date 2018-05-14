import sys
sys.path.append("/anaconda3/envs/udacity_tensorflow_py3/lib/python3.5/site-packages")
import pickle
import numpy as np
import math
import pandas as pd

from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Activation, Reshape, Embedding, concatenate
from keras.callbacks import ModelCheckpoint
import keras.backend as K
from keras import regularizers

class NN_with_embedding(object):

    def __init__(self, X_train, y_train, testing=False):
        self.epoch = 12
        self.X_train = X_train
        self.y_train = y_train
        self.max_log_y = np.max(np.log(self.y_train))
        self.testing = testing
        self.checkpointer = ModelCheckpoint(filepath="best_model_weights.hdf5", verbose=1, save_best_only=True)
        self.keras_model()

    def log_target(self,y):
        return np.log(y) / self.max_log_y

    def reverse_log_target(self,y):
        result = K.exp(y * self.max_log_y)
        return result

    def keras_model(self):
        #store ID is from 1 to 1115, instead of 0 to 1114, so add 1 here to include the integer "1115"
        input_store = Input(shape=(1,),name="store")
        model_store = Embedding(1115+1, 10, input_length=1)(input_store)
        output_store = Reshape(target_shape=(10,))(model_store)

        input_dow = Input(shape=(1,),name="dow")
        model_dow = Embedding(7+1, 1, input_length=1)(input_dow)
        output_dow = Reshape(target_shape=(1,))(model_dow)

        input_year = Input(shape=(1,),name="year")
        model_year = Embedding(3, 1, input_length=1)(input_year)
        output_year = Reshape(target_shape=(1,))(model_year)

        input_month = Input(shape=(1,),name="month")
        model_month = Embedding(12+1, 1, input_length=1)(input_month)
        output_month = Reshape(target_shape=(1,))(model_month)

        input_day = Input(shape=(1,),name="day")
        model_day = Embedding(31+1, 5, input_length=1)(input_day)
        output_day = Reshape(target_shape=(5,))(model_day)

        input_promo = Input(shape=(1,),name="promo")
        output_promo = Dense(1, input_dim=1)(input_promo)

        input_stateholiday = Input(shape=(1,),name="state_holiday")
        model_stateholiday = Embedding(4, 1, input_length=1)(input_stateholiday)
        output_stateholiday = Reshape(target_shape=(1,))(model_stateholiday)

        input_school = Input(shape=(1,),name="school_holiday")
        output_school = Dense(1, input_dim=1)(input_school)

        input_storetype = Input(shape=(1,),name="store_type")
        model_storetype = Embedding(4, 1, input_length=1)(input_storetype)
        output_storetype = Reshape(target_shape=(1,))(model_storetype)

        input_assortment = Input(shape=(1,),name="assortment")
        model_assortment = Embedding(3, 1, input_length=1)(input_assortment)
        output_assortment = Reshape(target_shape=(1,))(model_assortment)

        input_distance = Input(shape=(1,),name="comp_distance")
        output_distance = Dense(1, input_dim=1)(input_distance)

        input_competemonths = Input(shape=(1,),name="comp_since_total_month")
        model_competemonths = Embedding(25, 2, input_length=1)(input_competemonths)
        output_competemonths = Reshape(target_shape=(2,))(model_competemonths)

        input_promo2weeks = Input(shape=(1,),name="promo_since_total_week")
        model_promo2weeks = Embedding(29, 2, input_length=1)(input_promo2weeks)
        output_promo2weeks = Reshape(target_shape=(2,))(model_promo2weeks)
        
        input_promo2 = Input(shape=(1,),name="promo2")
        output_promo2 = Dense(1, input_dim=1)(input_promo2)

        input_latestpromo2months = Input(shape=(1,),name="promo_since_last_interval")
        model_latestpromo2months = Embedding(4, 1, input_length=1)(input_latestpromo2months)
        output_latestpromo2months = Reshape(target_shape=(1,))(model_latestpromo2months)

        input_promointerval = Input(shape=(1,),name="promo_interval")
        model_promointerval = Embedding(4, 1, input_length=1)(input_promointerval)
        output_promointerval = Reshape(target_shape=(1,))(model_promointerval)

        input_states = Input(shape=(1,),name="states")
        model_states = Embedding(12, 6, input_length=1)(input_states)
        output_states = Reshape(target_shape=(6,))(model_states)

        input_weekofyear = Input(shape=(1,),name="week_of_year")
        output_weekofyear = Dense(1, input_dim=1)(input_weekofyear)

        input_backwardopen = Input(shape=(1,),name="backward_open")
        model_backwardopen = Embedding(16, 1, input_length=1)(input_backwardopen)
        output_backwardopen = Reshape(target_shape=(1,))(model_backwardopen)

        input_forwardopen = Input(shape=(1,),name="forward_open")
        model_forwardopen = Embedding(16, 1, input_length=1)(input_forwardopen)
        output_forwardopen = Reshape(target_shape=(1,))(model_forwardopen)

        input_backwardpromo = Input(shape=(1,),name="backward_promo")
        model_backwardpromo = Embedding(16, 1, input_length=1)(input_backwardpromo)
        output_backwardpromo = Reshape(target_shape=(1,))(model_backwardpromo)

        input_forwardpromo = Input(shape=(1,),name="forward_promo")
        model_forwardpromo = Embedding(16, 1, input_length=1)(input_forwardpromo)
        output_forwardpromo = Reshape(target_shape=(1,))(model_forwardpromo)

        input_backwardschool = Input(shape=(1,),name="backward_school")
        model_backwardschool = Embedding(16, 1, input_length=1)(input_backwardschool)
        output_backwardschool = Reshape(target_shape=(1,))(model_backwardschool)

        input_forwardschool = Input(shape=(1,),name="forward_school")
        model_forwardschool = Embedding(16, 1, input_length=1)(input_forwardschool)
        output_forwardschool = Reshape(target_shape=(1,))(model_forwardschool)

        merged_embedding = concatenate([output_store, output_dow, output_year, output_month, output_day, output_promo, 
            output_stateholiday, output_school, output_storetype, output_assortment, output_distance, output_competemonths, 
            output_promo2weeks, output_promo2, output_latestpromo2months, output_promointerval, output_states, output_weekofyear,
            output_backwardopen, output_forwardopen, output_backwardpromo, output_forwardpromo, output_backwardschool, output_forwardschool])

        tensor = Dropout(0.03)(merged_embedding)
        tensor = Dense(1000, activation='relu')(tensor)
        tensor = Dense(500, activation='relu')(tensor)
        output = Dense(1, activation='sigmoid')(tensor)

        self.model = Model(inputs=[input_store, input_dow, input_year, input_month, input_day, input_promo, input_stateholiday, 
            input_school, input_storetype, input_assortment, input_distance, input_competemonths, input_promo2weeks, 
            input_promo2, input_latestpromo2months, input_promointerval, input_states, input_weekofyear, input_backwardopen, 
            input_forwardopen, input_backwardpromo, input_forwardpromo, input_backwardschool, input_forwardschool], outputs=[output])

        self.model.compile(loss='mean_absolute_error', optimizer='adam')

    def fit(self):

        if not self.testing:
            hist = self.model.fit(self.X_train, self.log_target(self.y_train),
                              epochs=self.epoch, batch_size=128,validation_split=0.02,
                           callbacks=[self.checkpointer],
                           )
            self.model.load_weights('best_model_weights.hdf5')
        else:
            hist = self.model.fit(self.X_train, self.log_target(self.y_train), epochs=self.epoch, batch_size=128)
        self.hist = hist
