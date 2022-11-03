import json
from sklearn.utils import shuffle
from numpy import loadtxt
from numpy import array, loadtxt
from sklearn.model_selection import KFold
from keras.models import Sequential, load_model
from keras.layers import Dense

kf = KFold(n_splits=5)

all_datas=[]

classes = {'agachamento':[0,0,1], 'extensaoquadril': [0,1,0], 'flexaojoelho': [1,0,0]}

f = open('all_angles.json')
data = json.load(f)

#data=convertNoiseToDetection(data)

count = 1
for label in data:
    for features in data[label]:
        specific_values=[]
        for vl in features:
            specific_values.append(vl)
        specific_values.append(label)
        all_datas.append(specific_values)
            

all_datas=shuffle(all_datas, random_state=0)

X=[]
y=[]

for i in all_datas:
    content_values=[]
    for j in range(5):
        if j!=4:
            content_values.append(i[j])
        else:
            y.append(classes[i[j]])
    X.append(content_values)

X_arr = array(X)
y_arr = array(y)

for train_indexes, test_indexes in kf.split(X):
    X_train = X_arr[train_indexes]
    y_train = y_arr[train_indexes]

    X_test = X_arr[test_indexes]
    y_test = y_arr[test_indexes]

    model = Sequential()
    model.add(Dense(10, input_dim=4, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(3, activation='softmax'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=250, batch_size=8)

    _, accuracy = model.evaluate(X_train, y_train)
    print('Accuracy: %.2f' % (accuracy*100))


    predictions = (model.predict(X_test) > 0.5).astype(int)

    write_detection=[]

    for i in range(len(X_test)):
        if(predictions[i][0] == 0 and predictions[i][1] == 0 and predictions[i][2] == 1):   
            write_detection.append([i,"agachamento"])
        elif(predictions[i][0] == 0 and predictions[i][1] == 1 and predictions[i][2] == 0):   
            write_detection.append([i,"extensaoquadril"])
        elif(predictions[i][0] == 1 and predictions[i][1] == 0 and predictions[i][2] == 0):   
            write_detection.append([i,"flexaojoelho"])    
        else:   
            write_detection.append([i,"no_detection"])

    with open("output_"+str(count), 'w') as f:
        json.dump(write_detection, f)
        
    model.save("detectionModel"+str(count)+"acuracia"+str(accuracy*100))
    count+=1
