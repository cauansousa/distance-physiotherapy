from json import dump, load
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
from numpy import array, loadtxt
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense

kf = KFold(n_splits=5)


Xt=[]
yt=[]

f = open('noise_data.json')
base = load(f)
f.close()


X = []
y = []
for i in base:
    input_values=[]
    output_values=[]
    for k in i[0]:
        input_values.append(k)
    input_values.append(i[1])
    input_values.append(i[2])
    input_values.append(i[3])
    input_values.append(i[4])
    for k in i[5]:
        output_values.append(k)
    X.append(input_values)
    y.append(output_values)


X_arr = array(X)
y_arr = array(y)

#f = open('all_features_noisy_original.json')
#base_test = load(f)
#f.close()



pr=int(len(base)*0.2)

data_test=[]
data=[]
#10% da base de dados para teste (como possuo poucos dados na base, optei por
#pegar uma parcela pequena)


# for count in range(0, 5):
    # for i in range(0,pr,1):
    #     data_test.append(base[i])

    # for i in range(pr,len(base),1):
    #     data.append(base[i])



    # for i in data:
    #     input_values=[]
    #     output_values=[]
    #     for k in i[0]:
    #         input_values.append(k)
    #     input_values.append(i[1])
    #     input_values.append(i[2])
    #     input_values.append(i[3])
    #     input_values.append(i[4])
    #     for k in i[5]:
    #         output_values.append(k)
    #     X.append(input_values)
    #     y.append(output_values)

    # for i in data_test:
    #     input_values=[]
    #     output_values=[]
    #     for k in i[0]:
    #         input_values.append(k)
    #     input_values.append(i[1])
    #     input_values.append(i[2])
    #     input_values.append(i[3])
    #     input_values.append(i[4])
    #     for k in i[5]:
    #         output_values.append(k)
    #     Xt.append(input_values)
    #     yt.append(output_values)

# X_train = []
# X_test = []
# y_train = []
# y_test = []

for train_indexes, test_indexes in kf.split(X):
    X_train = X_arr[train_indexes]
    y_train = y_arr[train_indexes]

    X_test = X_arr[test_indexes]
    y_test = y_arr[test_indexes]

    # with open("input_measure_x.json", 'w') as f:
    #     dump(X_test, f)
    # with open("output_measure_y.json", 'w') as f:
    #     dump(y_test, f) 

    model = Sequential()
    model.add(Dense(10, input_dim=7, activation='relu'))
    model.add(Dense(12, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(12, activation='relu'))
    model.add(Dense(13, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(6, activation='softmax'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    #epochs=250
    #model.fit(X, y, epochs=120, batch_size=5)
    model.fit(X_train, y_train, epochs=120, batch_size=5)

    _, accuracy = model.evaluate(X, y)
    print('Accuracy: %.2f' % (accuracy*100))

    model.save("measureModel3")


# def test():
    # model = load_model('measureModel3')

    # name_of_file="output_measure_x"
    # name_of_file_measure="output_measure_x"
    # name_of_file_detection="output_detection_x"
    # f = open(name_of_file_detection+'.json')
    # data_loaded = load(f)
    # f.close()

    # f = open('output_'+name_of_file_detection+'.json')
    # data_labeled = load(f)
    # f.close()

    one_result=[]


    classes = {'agachamento':[0,0,1], 'extensaoquadril': [0,1,0], 'flexaojoelho': [1,0,0], 'no_detection': [0,0,0]}

    data_loaded = X_test
    data_labeled = y_test

    index=0
    for g in data_loaded:
        aux=[]
        for fr in classes[data_labeled[index][1]]:
            aux.append(fr)

        index=index+1
        for f in g:
            aux.append(f)
        one_result.append(aux)    


    predictions = (model.predict(one_result) > 0.7).astype(int)

    output_array=[]

    for item in range(len(one_result)):
        content_array=[]
        content_array.append([1,0,0])
        for it in one_result[item]:
            content_array.append(it)
        aux=[]
        for vl in predictions[item]:
            aux.append(int(vl))
        content_array.append(aux)    
        output_array.append(content_array)


    # with open("output_measure_"+name_of_file+".json", 'w') as f:
    #     dump(output_array, f)

    #model.save("measureModel")

    number_of_true=0
    number_of_false=0
    for i in range(50):
        dif=0
        for j in range(4):
            if predictions[i][j] != yt[i][j]:
                dif=1
                break
        if dif:
            number_of_false+=1
        else:
            number_of_true+=1

        
    print(number_of_true)
    print(number_of_false)
    print(100*number_of_true/(number_of_false+number_of_true))



# test()
