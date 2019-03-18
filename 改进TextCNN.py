# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 10:06:23 2019

@author: Administrator
"""

data = pd.read_csv('总数据.csv',encoding='ISO-8859-1')
review = data['text']
label = data['label']
def review_to_words(raw_review):
    review_re = re.sub('[^a-zA-Z]',' ',raw_review) 
    review_lower = review_re.lower()
    review_split = review_lower.split()
    Stem = WordNetLemmatizer()
    review_stem = [Stem.lemmatize(word) for word in review_split]
    stops = set(stopwords.words('english'))
    review_afterstop = [w for w in review_stem if not w in stops]
    return(' '.join(review_afterstop))
all_review = [] 
for i in range(len(review)):
    if ((i+1)%5000 == 0):
        print('已处理第%d/%d条数据' % ((i+1),len(review)))
    all_review.append(review_to_words(review[i]))
    

words=[]
for raw_sentence in all_review:
    words.append(raw_sentence.split())

word_to_ix = {}
for sent  in words:
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
max_features = len(word_to_ix)

tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(all_review))
all_review = tokenizer.texts_to_sequences(all_review)
maxlen = 15
all_review= pad_sequences(all_review, maxlen=maxlen)

sentiment_score = np.array(data['Ãô¸ÐÖµµÃ·Ö'])
size = np.array(data['size'])
tag1 = np.array(data['tag1'])
tag2 = np.array(data['tag2'])
first = np.array(data['first'])
second = np.array(data['second'])
neg = np.array(data['neg'])
pos = np.array(data['pos'])
senti = np.array(data['senti'])

all_data = np.column_stack((all_review,sentiment_score,size,tag1,tag2,first,second,neg,pos,senti)) #增加属性

input_data,input_test,y_train,y_test = train_test_split(all_data,label,random_state=0,test_size=0.2) 

train_X = input_data[:,0:15]
test_X = input_test[:,0:15]
train_X_rest = input_data[:,15:24]
test_X_rest = input_test[:,15:24]

embed_size = 300
maxlen = 15

def Text_CNN():

    filter_sizes = [2,3] #不同size的filter
    num_filters = 20 #同一size filter的数量

    inp = Input(shape=(maxlen,))

    #embedding层
    x = Embedding(max_features,embed_size)(inp) #non-static模式embedding
    x = Reshape((maxlen,embed_size,1))(x)

    maxpoolList=[]

    #卷积层和pooling层
    for i in range(len(filter_sizes)):
        conv = Conv2D(num_filters,kernel_size=(filter_sizes[i],embed_size),
                      kernel_initializer='he_normal',activation='elu')(x)
        maxpoolList.append(MaxPool2D(pool_size=(maxlen-filter_sizes[i]+1,1))(conv))

    #连接pooling层输出的结果
    z = Concatenate(axis=1)(maxpoolList)
    z = Flatten()(z)

    #dropout正则化
    z = Dropout(0.5)(z)

    #全连接层
    y = Dense(1,activation="sigmoid")(z)


    model = Model(inputs=inp,outputs=y)

   
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

    return model
    
model = Text_CNN()
model.summary()
history = model.fit(train_X,y_train,epochs=10,batch_size=128,validation_split=0.2)
pred_y_test = model.predict(test_X, batch_size=1024, verbose=1)
print("recall score: {}".format(metrics.recall_score(y_test,(pred_y_test>0.5))))
print("accuracy score: {}".format(metrics.accuracy_score(y_test,(pred_y_test>0.5))))
print("F1 score: {}".format(metrics.f1_score(y_test,(pred_y_test>0.5))))

dense_layer_model = Model(inputs=model.input,outputs=model.get_layer('flatten_1').output)
dense_output = dense_layer_model.predict(train_X)
train = np.column_stack((dense_output,train_X_rest))
dense_output_test = dense_layer_model.predict(test_X)
test = np.column_stack((dense_output_test,test_X_rest))

model4 = models.Sequential() 
model4.add(layers.Dense(32, activation='relu', input_shape=(49,))) 
model4.add(layers.Dense(1, activation='sigmoid'))
model4.summary()
model4.compile(optimizer='rmsprop',loss='binary_crossentropy', metrics=['accuracy'])
history = model4.fit(train,y_train,epochs=10,batch_size=128,validation_split=0.2)
pred_y_test = model4.predict(test, batch_size=1024, verbose=1)
print("recall score: {}".format(metrics.recall_score(y_test,(pred_y_test>0.5))))
print("accuracy score: {}".format(metrics.accuracy_score(y_test,(pred_y_test>0.5))))
print("F1 score: {}".format(metrics.f1_score(y_test,(pred_y_test>0.5))))