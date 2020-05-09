## kaggle用ツール

**1. 特徴量管理(FE_management)**

特徴量を列ごとにpickleで管理するもの,Datasetクラスに基本機能を記述。featureクラスには特徴量作成文を記載する。


**2. 学習進捗管理(base_trainmodel)**

data,model,config,features,notebook,logsで構成される環境を用意する。dataにベースとなる特徴を入れる。そこに追加したい特徴量をfeaturesから選択
学習済みmodelをmodelに保存、各種パラメータをjson形式でcinfigに保存、logsにfiやスコアを保存する。
