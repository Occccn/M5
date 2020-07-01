import sys
import pandas as pd
import numpy as np
import os
import lightgbm as lgb
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import json
import importlib

args = sys.argv

day_number = int(args[1])
store_number = int(args[2])

class groupgby_dataset():
    def __init__(self,category = None,category_num = None,category_2nd = None,category_num_2nd = None):
        self.dir = '../features/'
        with open('../data/data_full.joblib', mode="rb") as f:
            self.data = joblib.load(f)
        self.idx = self.data[self.data[category] == category_num].index
        self.data = self.data.iloc[self.idx]
        
        if category_2nd != None:
            self.idx = self.data[self.data[category_2nd] == category_num_2nd].index
            self.data = self.data.iloc[self.idx]
        

    
            
    def get_features(self,features = None ,path = None):

    #作成した特徴量の取得
        if features == None:
            print('features not selected')
            exit(0)
        else:
            dfs = []
            for feature in features:
                with open(path +'/'+ feature + '.joblib', mode="rb") as f:
                    tmp = joblib.load(f)
                    dfs.append(tmp.iloc[self.idx])
            tmp = pd.concat(dfs, axis=1)
            self.data = pd.concat([self.data,tmp],axis=1)

        return self.data
    
    def drop_features(self,features = None):
        self.data = self.data.drop(columns = features)
    


class lightgbm():
    def __init__(self ,data):
        self.data = data
        
    def get_params(self,params):
        self.params = params
        
    def get_features(self , features):
        self.features = features
        
    def get_train_test(self):
        #self.data_train = self.data[(self.data['part'] == 'train') ]
        self.data_train = self.data[(self.data['part'] == 'train') ]
        #self.data_test = self.data[self.data['part'] == 'test2']
    
    
    def hold_out(self):
        END_TRAIN   = 1941               # End day of our train set
        P_HORIZON   = 28  
        self.get_train_test()
        x_val = self.data_train[self.data_train['d'] > (END_TRAIN-P_HORIZON)]
        y_val = self.data_train[self.data_train['d'] > (END_TRAIN-P_HORIZON)]['demand']
        x_train = self.data_train[self.data_train['d'] <= (END_TRAIN-P_HORIZON)] 
        y_train = self.data_train[self.data_train['d'] <= (END_TRAIN-P_HORIZON)]['demand']
        train_set = lgb.Dataset(x_train[self.features], y_train)
        val_set = lgb.Dataset(x_val[self.features], y_val)
        del x_val ,y_val ,x_train ,y_train
        return train_set , val_set
    

    def cv(self , cv_num):
        END_TRAIN   = 1941               # End day of our train set　　1941
        P_HORIZON   = cv_num
        self.get_train_test()
        x_val = self.data_train[(self.data_train['d'] > (END_TRAIN-P_HORIZON)) & (self.data_train['d'] <= (END_TRAIN-P_HORIZON + 28)) ]
        y_val = self.data_train[(self.data_train['d'] > (END_TRAIN-P_HORIZON)) & (self.data_train['d'] <= (END_TRAIN-P_HORIZON + 28)) ]['demand']
        x_train = self.data_train[self.data_train['d'] <= (END_TRAIN-P_HORIZON)] 
        y_train = self.data_train[self.data_train['d'] <= (END_TRAIN-P_HORIZON)]['demand']
        train_set = lgb.Dataset(x_train[self.features], y_train)
        val_set = lgb.Dataset(x_val[self.features], y_val)
        del x_val ,y_val ,x_train ,y_train
        return train_set , val_set
    
    
    def valid_fit(self,hold_out = None , model_name = 'sample', model_save = None ) :
        self.model_name = model_name
        if hold_out == True:
            
            train_set,val_set = self.hold_out()
            num_boost_round = 2500
            early_stopping_rounds = 50
            self.model = lgb.train(self.params, train_set, num_boost_round = num_boost_round, early_stopping_rounds = early_stopping_rounds, 
                  valid_sets = val_set, verbose_eval = 100, feval = self.data.wrmsse)
            

            
            self.best_score_ = self.model.best_score
            
            
            ##feature_importance
            self.feature_importances_ = self.model.feature_importance()
                                      
            fi_df = pd.DataFrame({'feature': self.features,'feature importance': self.feature_importances_}).sort_values('feature importance', ascending = False)
            plt.figure(figsize=(10,10))
            sns.barplot(fi_df['feature importance'],fi_df['feature'])
            plt.rcParams["font.size"] = 15
            plt.savefig('../logs/' + model_name + '_fi.png')
            
            
            ##config
            dictionary = {'features':self.features, 'params':self.params, 'cv':'hold_out' ,
              'num_boost_round':num_boost_round ,'early_stopping_rounds':early_stopping_rounds , 'num_trees':self.model.num_trees()}
            with open('../config/' + model_name +'.json' , 'w') as outfile:
                json.dump(dictionary, outfile)
                
                
            ##score(適当)
            with open('../logs/' + model_name + '.log', mode='w') as f:
                f.write(str(self.model.best_score))

            del train_set , val_set

        else:
            cv_scores = []
            num_trees = []
            cv_list = [28 ,56 , 84]
            for cv_num in cv_list:
                print('CV_{0} ~ {1}days'.format(1941 - cv_num , 1941 - cv_num + 28))
                train_set,val_set = self.cv(cv_num)
                num_boost_round = 2500
                early_stopping_rounds = 50
                self.model = lgb.train(self.params, train_set, num_boost_round = num_boost_round, early_stopping_rounds = early_stopping_rounds, 
                      valid_sets = val_set, verbose_eval = 100, feval = self.data.wrmsse)

                for i in self.model.best_score.values():
                    cv_scores.append(i['wrmsse'])
                num_trees.append(self.model.num_trees())

            for i in range(len(cv_scores)):
                print('CV{0}_WRMSSE_SCORE : {1}'.format(i+1 , cv_scores[i]))
            print('CV_mean_WRMSSE_SCORE : {0}'.format(np.mean(cv_scores)))
            ##score(CVの各スコアを保存)
            with open('../logs/' + model_name + '.log', mode='w') as f:
                f.write(str(cv_scores))


            ##config(predict用のモデルを作成する際にハイパラを利用するために保存、num_treesは各CVの平均としている)
            dictionary = {'features':self.features, 'params':self.params, 'cv':'hold_out' ,
              'num_boost_round':num_boost_round ,'early_stopping_rounds':early_stopping_rounds , 'num_trees':int(np.mean(num_trees)) , 'CV_mean_score':np.mean(cv_scores)}
            with open('../config/' + model_name +'.json' , 'w') as outfile:
                json.dump(dictionary, outfile)



    def fit_for_predict(self , valid_name =None):

      num_boost_round = 350
      self.params['n_estimators'] = 350

      #train_data作成、test1の正解追加後に変更必須
      data_train = self.data[self.data['part'] == 'train']
      train_set = lgb.Dataset(data_train[self.features], data_train['demand'])
      self.model = lgb.train(self.params, train_set, num_boost_round = num_boost_round)

      filename =  '../Model/'+ valid_name  + '.joblib'
      joblib.dump(self.model, filename)


    def load_model(self , store , day):
      with open('../Model/' + 'store{0} _day{1}'.format(store , day) + '.joblib', mode="rb") as f:
          self.model = joblib.load(f)


##再帰モデルのまま    
    def predict(self ,data , day ,  part = 'test2'):
      
        tmp = self.model.predict(self.data[self.data['d'] ==1942 + int(day) -1 ][self.features])

        #ループの日付のdemand部分を置き換える
        self.data.loc[self.data[self.data['d'] ==i].index , 'demand'] = tmp

        filename = '../predict/' + self.model_name +'_' + part + '.joblib' 
        joblib.dump(self.data[self.data['part'] == 'test1']['demand'],filename)
                              
data = groupgby_dataset(category='store_id' , category_num= store_number)


##SNAP_lag
path = '../features/SNAP_Feature'
if store_number <= 3:
    SNAP_lag_features = ['Lag_SNAP_CA']
if (store_number > 3) and (store_number <= 6):
    SNAP_lag_features = ['Lag_SNAP_TX']
if (store_number > 6):
    SNAP_lag_features = ['Lag_SNAP_WI']
data.get_features(features=SNAP_lag_features , path = path)
        

##rolling
path = '../features/rolling_lag_Feature'
demand_lag_features = ['rolling_lag_mean_t7', 'rolling_lag_mean_t28', 'rolling_lag_mean_t56' ,'rolling_lag_mean_t84' ,'rolling_lag_mean_t168',
 'rolling_lag_std_t7' ,'rolling_lag_std_t28' , 'rolling_lag_std_t56',  'rolling_lag_std_t84' ,'rolling_lag_std_t168' , 
'rolling_lag_max_t7', 'rolling_lag_max_t28',  'rolling_lag_max_t56', 
'rolling_lag_min_t7','rolling_lag_min_t28','rolling_lag_min_t56']
for i in range(len(demand_lag_features)):
    demand_lag_features[i] = demand_lag_features[i] + '_shift' + str(day_number)
data.get_features(features=demand_lag_features , path = path)


#mean_encoding
path = '../features/Ordered_TS_mean_encoding'
Ordered_TS_features = ['Ordered_TS_id' , 'Ordered_TS_id_price']
for i in range(len(Ordered_TS_features)):
    Ordered_TS_features[i] = Ordered_TS_features[i] + '_shift' + str(day_number)
data.get_features(features=Ordered_TS_features, path = path)

#price_features
path = '../features/price_feature'
price_feature = ['rolling_price_mean_t7'	,'rolling_price_mean_t28'	,'rolling_price_mean_t56',	'rolling_price_mean_t84'	,'rolling_price_mean_t168'	,
                 'rolling_price_std_t7'	,'rolling_price_std_t28'	,'rolling_price_std_t56'	,'rolling_price_std_t84'	,'rolling_price_std_t168'	,
                 'price_max' ,'price_min' ,'price_mean' ,'price_norm' ,'cnt_price_change_up' ,'price_change_up_lag','cnt_price_change_down' ,'price_change_down_lag']
data.get_features(features=price_feature, path = path)

#calendar
path = '../features/calendar_feature'
calendar_feature = ['day2','week' , 'month', 'year' , 'quarter' , 'dayofweek' , 'is_month_end' , 'is_month_start'  , 'is_quarter_end' , 'is_quarter_start' , 'is_year_end' , 'is_year_start' ]
data.get_features(features=calendar_feature, path = path)

#etc_features
path = '../features/etc_features'
etc_feature = ['demand_min_lag' , 'demand_max_lag']
for i in range(len(etc_feature)):
    etc_feature[i] = etc_feature[i] + '_shift' + str(day_number)
etc_feature.append( 'item_firstday')
data.get_features(features=etc_feature, path = path)

#demand_probability
path = '../features/demand_probability'
demand_probability_feature = ['demand_0_prob' ,'demand_1_prob' ,'demand_2_prob' ,'demand_3_prob' ,'demand_4_prob' ,'demand_5_prob' ,
                              'demand_6_prob' ,'demand_7_prob' ,'demand_8_prob' ,'demand_9_prob' ,'demand_10_prob']
for i in range(len(demand_probability_feature)):
    demand_probability_feature[i] = demand_probability_feature[i] + '_shift' + str(day_number)
data.get_features(features=demand_probability_feature, path = path)

path = '../features/other_rolling_features_shiftX'
other_feature=["first_location_of_minimum_t168" ,  "first_location_of_minimum_t7" , "first_location_of_minimum_t28" ,  "first_location_of_minimum_t56" , 
                 "first_location_of_max_t168" , "first_location_of_max_t7" , "first_location_of_max_t28" , "first_location_of_maximum_t56" , 
                 "llmin_t7","llmin_t168",  "last_location_of_minimum_t28" ,  "last_location_of_minimum_t56" , 
                 "llman_t7" , "llman_t168" ,"last_location_of_maximum_t28" , "last_location_of_maximum_t56"  ]
for i in range(len(other_feature)):
    other_feature[i] = other_feature[i] + '_shift_' + str(day_number)
data.get_features(features=other_feature, path = path)

data.get_features(features=other_feature , path = path)



#lgb_params
lgb_params = {'boosting_type': 'gbdt','objective': 'tweedie','tweedie_variance_power': 1.1,
                    'metric': 'custom','subsample': 0.5,'subsample_freq': 1,'learning_rate': 0.03,'num_leaves': 2**11-1,
                    'min_data_in_leaf': 2**12-1,'feature_fraction': 0.5,'max_bin': 100,'n_estimators': 1400,'boost_from_average': False,'verbose': -1,
                } 




#features
base_features = [ "item_id","dept_id","cat_id","event_name_1","event_type_1","event_name_2","event_type_2", "snap_CA","sell_price"]
features = base_features + SNAP_lag_features + calendar_feature + price_feature + Ordered_TS_features  +demand_lag_features + etc_feature+ demand_probability_feature 




lgb_clf = lightgbm(data = data.data)
lgb_clf.get_params(params=lgb_params)
lgb_clf.get_features(features=features)
lgb_clf.fit_for_predict(valid_name = 'store{0}_day{1}'.format(store_number , day_number))