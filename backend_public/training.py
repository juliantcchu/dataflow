
from tensorflow import keras
from tensorflow.keras import layers as layers
import numpy as np
from model_class import modelClass
# from trainingCallback import *
import os
import datetime
import math

# from createDatasetTest import mnist
from fileSystemDb import fileSystemDb

DB = fileSystemDb()


def trainModelBackend(data):
    
    #create model from model data
    model_params = DB.getModelArchitecture(data['modelArchitecture'])
    model1 = modelClass(model_params)
    model = model1.createModel(print_summary=True)


    #define inputs and outputs
    ioObjects = data['ioObjects']
    print(ioObjects)
    inputs = {}
    outputs = {}
    output_names = model1.getOutputNames()
    print('loading datasets:')
    for ioObject in ioObjects.values():
        if ioObject['type'] == 'input':
            inputs[ioObject['name']] = DB.getDataObject(ioObject['dataset'])
        elif ioObject['type'] == 'output':
            name = output_names[ioObject['name']]
            outputs[name] = DB.getDataObject(ioObject['dataset'])

    # =================MNIST TEST===================================
    #x_train, y_train, x_test, y_test = mnist()
    # =================MNIST TEST===================================
    #callbackFunc = CustomCallback()

    training_params = data['trainingParams']
    model.compile(
        optimizer = training_params['optimizer'],
        loss = training_params['loss_function'], 
        metrics = training_params['metrics']
    )

    #startTrainingTime = datetime.datetime.now()

    history = model.fit(
        x = inputs, 
        y = outputs, 
        epochs = eval(training_params['epochs']),
        batch_size = eval(training_params['batch_size']), 
        validation_split=eval(training_params['val_split']),
        #callbacks = [callbackFunc]
    )

    #check for NaN
    for key in history.history.keys():
        history.history[key] = ['NaN' if math.isnan(x) else x for x in history.history[key]]

    datetimeNow = str(datetime.datetime.now())
    modelTrainingInfo = { 
        'modelArchitecture': data['modelArchitecture'],
        'datetime': datetimeNow,
        'trainingParams': data['trainingParams'],
        'trainingHistory': history.history, 
        'dataObjects': '(under development)',
        'trainingTime': '(under development)' 
    }
    # print(type(history.history), modelTrainingInfo)

    DB.saveTrainedModelWithInfo(data['modelArchitecture'] + '||' + datetimeNow, model, modelTrainingInfo)

    return  'success'



