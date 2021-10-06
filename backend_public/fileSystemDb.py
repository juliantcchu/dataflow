import os
import numpy as np
from tensorflow import keras
import json
import datetime
from shutil import copytree, rmtree


class fileSystemDb:

    def __init__(self):
        f = open('currLocation', 'r')
        self.location = f.read()
        f.close()

    def setLocation(self, location):
        if not os.path.isdir(location):
            #create the directory if it doesn't exist
            os.makedirs(os.path.join(location, 'dataObjects'))
            os.mkdir(os.path.join(location, 'modelArchitectures'))
            os.mkdir(os.path.join(location, 'trainedModels'))
        f = open('currLocation', 'w')
        f.write(location)
        f.close()
        self.location = location


    def saveJson(self, fileName, data):
        file = open(fileName, "w")
        json.dump(data, file)
        file.close()
    
    def readJson(self, fileName):
        file = open(fileName, "r")
        content = json.load(file)
        file.close()
        return content

    # def getDataPipeline(self, model_arch_name):
    #     self.checkValidName(model_arch_name)
    #     fileName = os.path.join(self.location, 'dataPipelines', model_arch_name)
    #     content = self.readJson(fileName)
    #     return content

    # def saveDataPipeline(self, model_arch_name, new_data):
    #     self.checkValidName(model_arch_name)
    #     fileName = os.path.join(self.location, 'dataPipelines', model_arch_name)
    #     self.saveJson(fileName, new_data)

    # def deleteDataPipeline(self, model_arch_name):
    #     self.checkValidName(model_arch_name)
    #     fileName = os.path.join(self.location, 'dataPipelines', model_arch_name)
    #     os.remove(fileName)

    # def listDataPipelines(self):
    #     return [f for f in os.listdir(os.path.join(self.location, 'dataPipelines')) if not f.startswith('.')]


    def checkValidName(self, name):
        # for security, edit
        return True

    def listModelArchitectures(self):
        return [f for f in os.listdir(os.path.join(self.location, 'modelArchitectures')) if not f.startswith('.')]

    def listTrainedModel(self):
        return [f for f in os.listdir(os.path.join(self.location, 'trainedModels')) if not f.startswith('.')]
    
    def getModelArchitecture(self, model_arch_name):
        self.checkValidName(model_arch_name)
        fileName = os.path.join(self.location, 'modelArchitectures', model_arch_name)
        print(fileName)
        content = self.readJson(fileName)
        return content

    def saveModelArchitecture(self, model_arch_name, new_data):
        self.checkValidName(model_arch_name)
        fileName = os.path.join(self.location, 'modelArchitectures', model_arch_name)
        self.saveJson(fileName, new_data)

    def deleteModelArchitecture(self, model_arch_name):
        self.checkValidName(model_arch_name)
        fileName = os.path.join(self.location, 'modelArchitectures', model_arch_name)
        os.remove(fileName)

    def getTrainedModel(self, trained_model_name):
        self.checkValidName(trained_model_name)
        fileName = os.path.join(self.location, 'trainedModels', trained_model_name, 'model')
        return keras.models.load_model(fileName)
    
    def exportTrainedModel(self, trained_model_name, directory):
        self.checkValidName(trained_model_name)
        fileName = os.path.join(self.location, 'trainedModels', trained_model_name, 'model')
        print('copying ' +fileName + ' to '+directory)
        copytree(fileName, directory)

    def saveTrainedModel(self, trained_model_name, model):
        self.checkValidName(trained_model_name)
        fileName = os.path.join(self.location, 'trainedModels', trained_model_name, 'model')
        model.save(fileName)
    
    def saveTrainedModelInfo(self, trained_model_name, info):
        #info = {training_acc, training_loss, val_acc, val_loss, model_rchitecture, dataObjects}
        self.checkValidName(trained_model_name)
        fileName = os.path.join(self.location, 'trainedModels', trained_model_name, 'info.tlog')
        self.saveJson(fileName, info)


    def saveTrainedModelWithInfo(self, trained_model_name, model, info): 
        self.saveTrainedModel(trained_model_name, model)
        self.saveTrainedModelInfo(trained_model_name, info)

    def getTrainedModelInfo(self, trained_model_name):
        self.checkValidName(trained_model_name)
        fileName = os.path.join(self.location, 'trainedModels', trained_model_name, 'info.tlog')
        return self.readJson(fileName)
    
    def deleteTrainedModel(self, trained_model_name):
        self.checkValidName(trained_model_name)
        fileName = os.path.join(self.location, 'trainedModels', trained_model_name)
        rmtree(fileName)
        # os.rmdir(fileName)
 
    def getDataObject(self, name):
        self.checkValidName(name)
        addr = os.path.join(self.location, 'dataObjects', name, 'data')
        return np.load(addr)

    def saveDataObject(self, name, data):
        self.checkValidName(name)
        directory = os.path.join(self.location, 'dataObjects', name)
        try:
            os.makedirs(directory)
        except FileExistsError:
            pass
        info_addr = os.path.join(directory, 'info')
        self.saveJson(info_addr, {'name': name, 'datetime': str(datetime.datetime.now())})
        data_addr = os.path.join(directory, 'data')
        data.save(data_addr)

    def deleteDataObjects(self, dataObjectName):
        self.checkValidName(dataObjectName)
        fileName = os.path.join(self.location, 'dataObjects', dataObjectName)
        try: 
            rmtree(fileName)
        except FileNotFoundError:
            return 'no such file'


    def listDataObjects(self):
        return [f for f in os.listdir(os.path.join(self.location, 'dataObjects')) if not f.startswith('.')]

    def listDataObjectsInfo(self):
        listOfDataObjects = self.listDataObjects()
        listOfDataObjects_info = []
        for dataObject in listOfDataObjects:
            listOfDataObjects_info.append(
                self.readJson(os.path.join(self.location, 'dataObjects', dataObject, 'info'))
            )
        return listOfDataObjects_info

    def getDataset(self, dataset_name):
        pass

    # def getData(self, name):
    #     pass