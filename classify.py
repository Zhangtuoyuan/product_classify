from tensorflow.keras.models import Model   # 匯入 Model 類別
from tensorflow.keras.layers import (Input, Dense, Embedding,
                LSTM, Conv2D, MaxPooling2D, Flatten, concatenate)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping

import pandas as pd
import numpy as np
import jieba
import util as u
import os
import math 
import category_encoders as ce


# directory = './output/'
directory = './product_BrandName_Classify/'

# -----------------
#  斷字斷詞
# -----------------
jieba.set_dictionary("product_BrandName_Classify/dict.txt.big.txt")
stop_words =set()
with open('product_BrandName_Classify/標點符號.txt','r',encoding='utf-8-sig') as f:
    stop_words = f.read().split('\n')

def Jieba_and_CleanData(unclear_data):
    words = jieba.lcut(unclear_data,cut_all = False)
    clear_data = [w for w in words if w not in stop_words and w != '\n']
    clear_data = " ".join(clear_data)
    return clear_data

# -----------------
# 資料載入
# -----------------

df = pd.read_csv('product_BrandName_Classify/train_data/TRAIN_通路單價品項.csv', error_bad_lines=False)
data_count = len(df.index)

# 輸入內容
pnames = pd.Series(df["product_name"])# 品名  string
prices = pd.Series(df["unit_price"])# 單價    float
channels = pd.Series(df["channel_id"])# 通路  int

# 預測內容
brands = pd.Series(df["brand_id"])# 品牌      int
c0s = pd.Series(df["category0"])# 大類        string
c3s = pd.Series(df["category3"])# 細類        string

pnames = pnames.apply(Jieba_and_CleanData)

# ----------------
#  建立標籤
# ----------------

# brand:品牌
brand_cat = brands.astype('category')
brand_cat_codes = brand_cat.cat.codes  # brands: 是 ID
brand_cat_codes =pd.get_dummies(brand_cat_codes,dtype=int) 
brand_labels = brand_cat_codes.to_numpy()
brand_labels_len =len(brand_cat.cat.categories)
# c0:大分類
c0_cat = c0s.astype('category')
c0_cat_codes = c0_cat.cat.codes
c0_cat_codes = pd.get_dummies(c0_cat_codes,dtype=int)
c0_labels = c0_cat_codes.to_numpy()
c0_labels_len =len(c0_cat.cat.categories)
# c3:細分類
c3_cat = c3s.astype('category')
c3_cat_codes = c3_cat.cat.codes
c3_cat_codes = pd.get_dummies(c3_cat_codes,dtype=int)
c3_labels = c3_cat_codes.to_numpy()
c3_labels_len =len(c3_cat.cat.categories)

print('brand',brand_labels_len)
print('c0',c0_labels_len)
print('c3',c3_labels_len)

# --------------
# 建立訓練模型
# --------------

def build_c0():
    # data
    tok = Tokenizer()
    tok.fit_on_texts(pnames)
    pnames_vec = tok.texts_to_sequences(pnames)
    maxlen_pd = max(map(len,pnames_vec))#找出最長句子的長度
    pnames_train_data = pad_sequences(pnames_vec,maxlen_pd)  #train_data 是產生完畢的訓練資料
    print(pnames_train_data[:10])
    pnames_vocab_size = len(tok.word_index) + 1
    # model
    pnames_in = Input(shape =(maxlen_pd,),dtype='int32')
    ecoded = Embedding(pnames_vocab_size,50)(pnames_in)
    ecoded = LSTM(100)(ecoded)
    out = Dense(c0_labels_len,activation='softmax')(ecoded)
    model = Model([pnames_in],out)
    model.compile(optimizer='rmsprop',
                    loss='categorical_crossentropy',
                    metrics=['acc'])

    history = model.fit(pnames_train_data
                    ,c0_labels
                    ,validation_split=0.05
                    ,batch_size=32
                    ,epochs=5 #20
                    ,verbose=2)

    model.save_weights(directory+"c0.h5")

    # --------------
    #  模型成效輸出
    # --------------
    u.plot(history.history,('acc','val_acc'),
                ' training&validation acc',('Epoch','Acc'))

def build_c1(c0):
    # 準備資料
    df_pnames = pnames[df["category0"]==c0]
    df_prices = prices[df["category0"]==c0]
    # label
    df_c1 = brands[df["category0"]==c0]
    c1_cat_c0 = df_c1.astype('category')
    c1_cat_codes_c0 = c1_cat_c0.cat.codes  # brands: 是 ID
    c1_cat_codes_c0 =pd.get_dummies(c1_cat_codes_c0,dtype=int) 
    c1_labels_c0 = c1_cat_codes_c0.to_numpy()
    c1_labels_len_c0 =len(c1_cat_c0.cat.categories)

    # 價格訓練資料處理
    max_price = df_prices.max()
    df_prices = df_prices / max_price
    prices_train_data = df_prices.to_numpy()

    # 品項訓練資料處理（產生詞向量）
    tok = Tokenizer()
    tok.fit_on_texts(pnames)
    pnames_vec = tok.texts_to_sequences(df_pnames) # 句子轉為sequence
    maxlen_pd = max(map(len,pnames_vec))           # 找出最長句子的長度
    pnames_train_data = pad_sequences(pnames_vec,maxlen_pd)  #轉為訓練資料
    pnames_vocab_size = len(tok.word_index) + 1    # 取得 不同字彙總數量
    
    # 通路訓練資料處理

    df_channels = channels[df["category0"]==c0]
    df_channels_types = df_channels.unique()
    print(df_channels_types)

    df_channels =  pd.DataFrame(df_channels)
    encoder = ce.BinaryEncoder(cols=['channel_id']).fit(df_channels)
    channels_dataset = encoder.transform(df_channels)
    channels_train_data = channels_dataset.to_numpy()
    # print(channels_train_data[:10])

    channels_maxlen = channels_train_data.shape[-1]
    # print(channels_train_data.shape,channels_maxlen)

    # 品項模型
    pnames_input = Input(shape =(maxlen_pd,),dtype='int32')
    pnames_src = Embedding(pnames_vocab_size,50)(pnames_input)
    pnames_src = LSTM(100)(pnames_src) #100個神經元
    pnames_src = Dense(128,activation='relu')(pnames_src) #100個神經元
    
    #單價模型
    prices_input = Input(shape=(1,), name='price')
    prices_src = Dense(8, activation='relu')(prices_input)

    # 連階層
    out = concatenate([pnames_src, prices_src], axis=-1) # 用輔助函式串接 3 個張量
    out = Dense(c1_labels_len_c0,activation='softmax')(out)

    model = Model([pnames_input,prices_input],out)
    
    model.compile(optimizer='rmsprop',
                    loss='categorical_crossentropy',
                    metrics=['acc'])

    history = model.fit([pnames_train_data,channels_train_data,prices_train_data]
                    ,c1_labels_c0
                    ,validation_split=0.05
                    ,batch_size=128
                    ,epochs= 20
                    ,verbose=2)

    file_name = directory+'c1-{}'.format(c0.replace('/','-'))
    model.save_weights(file_name+'.h5')

    # --------------
    #  模型成效輸出
    # --------------
    u.plot(history.history,('acc','val_acc'),
                '{}-training&validation acc'.format(c0),('Epoch','Acc'),file_name=file_name+'.png')

def build_brand(c0,file_name):
    # 準備資料
    df_pnames = pnames[df["category0"]==c0]
    train_data_count = len(df_pnames.index)
    if train_data_count < 50:
        return
    df_prices = prices[df["category0"]==c0]
    # label
    df_brands = brands[df["category0"]==c0]
    brand_cat_c0 = df_brands.astype('category')
    brand_cat_codes_c0 = brand_cat_c0.cat.codes  # brands: 是 ID
    brand_cat_codes_c0 =pd.get_dummies(brand_cat_codes_c0,dtype=int) 
    brand_labels_c0 = brand_cat_codes_c0.to_numpy()
    brand_labels_len_c0 =len(brand_cat_c0.cat.categories)

    # 價格訓練資料處理
    max_price = df_prices.max()
    df_prices = df_prices / max_price
    prices_train_data = df_prices.to_numpy()

    # 品項訓練資料處理（產生詞向量）
    tok = Tokenizer()
    tok.fit_on_texts(pnames)
    pnames_vec = tok.texts_to_sequences(df_pnames) # 句子轉為sequence
    maxlen_pd = max(map(len,pnames_vec))           # 找出最長句子的長度
    pnames_train_data = pad_sequences(pnames_vec,maxlen_pd)  #轉為訓練資料
    pnames_vocab_size = len(tok.word_index) + 1    # 取得 不同字彙總數量
    
    # 通路訓練資料處理

    df_channels = channels[df["category0"]==c0]
    df_channels_types = df_channels.unique()
    # print(df_channels_types)

    df_channels =  pd.DataFrame(df_channels)
    encoder = ce.BinaryEncoder(cols=['channel_id']).fit(df_channels)
    channels_dataset = encoder.transform(df_channels)
    channels_train_data = channels_dataset.to_numpy()
    # print(channels_train_data[:10])

    channels_maxlen = channels_train_data.shape[-1]
    # print(channels_train_data.shape,channels_maxlen)

    # 品項模型
    pnames_input = Input(shape =(maxlen_pd,),dtype='int32')
    pnames_src = Embedding(pnames_vocab_size,50)(pnames_input)
    pnames_src = LSTM(100)(pnames_src) #100個神經元
    pnames_src = Dense(128,activation='relu')(pnames_src) #100個神經元

    #通路模型
    input_dim_channels_count = len(df_channels_types)
    embedding_size = min(50,(input_dim_channels_count + 1) // 2)
    
    channels_input = Input(shape=(channels_maxlen,))
    channels_src = Embedding(input_dim = input_dim_channels_count, output_dim = embedding_size)(channels_input) #name="embedding"
    channels_Fla = Flatten()(channels_src)
    channels_src = Dense(8, activation='relu')(channels_Fla)
    
    #單價模型
    prices_input = Input(shape=(1,), name='price')
    prices_src = Dense(8, activation='relu')(prices_input)

    # 連階層
    out = concatenate([pnames_src, channels_src, prices_src], axis=-1) # 用輔助函式串接 3 個張量
    out = Dense(brand_labels_len_c0,activation='softmax')(out)

    model = Model([pnames_input,channels_input,prices_input],out) 
    model.compile(optimizer='rmsprop',
                    loss='categorical_crossentropy',
                    metrics=['acc'])

    early_stopping = EarlyStopping(
                        monitor='val_acc'
                        ,min_delta=0.05
                        ,patience=10
                        , verbose=1)

    history = model.fit([pnames_train_data,channels_train_data,prices_train_data]
                    ,brand_labels_c0
                    ,validation_split=0.2 #0.05
                    ,callbacks=[early_stopping]
                    ,batch_size=128
                    ,epochs= 50
                    ,verbose=2)

    path = directory+'brand-{}'.format(file_name)
    model.save_weights(path+'.h5')

    # --------------
    #  模型成效輸出
    # --------------
    u.plot(history.history,('acc','val_acc'),
                '{}-training&validation acc (data={})'.format(c0,train_data_count),('Epoch','Acc'),file_name=path+'.png')
    
# 主程序控制

def build_all_brand():
    c0_list = c0_cat.values.categories.to_list()
    for c0 in c0_list:
        print('--- build category brand: {} ---'.format(c0))
        file_name = c0.replace('/','-')
        if not os.path.exists(directory+'brand-{}.h5'.format(file_name)):
            build_brand(c0,file_name)

if not os.path.exists(directory+'c0.h5'):
    build_c0()

build_all_brand()

print('模型建立完畢！！！')

# label_object['Product'].inverse_transform(df['Product'])

