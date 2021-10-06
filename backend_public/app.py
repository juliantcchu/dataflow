#test-test-1
#test2
#test comment 3
#test4
#test-comment(2)
from flask import Flask
from flask_restful import Api, Resource, reqparse
from flask_cors import CORS
import os
import json
from training import trainModelBackend
# from dataPipeline import runDataPipelineBackend
from fileSystemDb import fileSystemDb

import numpy as np
import tensorflow as tf
from tensorflow import keras
import werkzeug

app = Flask(__name__)
app.secret_key = 'DUMMY_KEY' #os.getenv('FLASK_SECRET_KEY')
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1


api = Api(app)
CORS(app)

DB = fileSystemDb()

# change directory
changeProject_args = reqparse.RequestParser()
changeProject_args.add_argument('directory', type = str)
class changeProject(Resource):
    def post(self): 
        directory = changeProject_args.parse_args().directory
        if directory:
            DB.setLocation(directory)
        return DB.location
api.add_resource(changeProject, '/changeProject')

# train model
trainModel_args = reqparse.RequestParser()
trainModel_args.add_argument('data', type = str)
class trainModel(Resource):
    def post(self): 
        try: 
            args = trainModel_args.parse_args()
            data = json.loads(args.data)
            
            result = trainModelBackend(data)
            return result #{'results': eval}
        except Exception as e:
            print(e)
            return str(e)

api.add_resource(trainModel, '/trainModel')

# # run Data Pipeline
# runDataPipeline_args = reqparse.RequestParser()
# runDataPipeline_args.add_argument('data', type = str)
# class runDataPipeline(Resource):
#     def post(self): 
#         try: 
#             args = runDataPipeline_args.parse_args()
#             data = json.loads(args.data)
#             print('\n\n\ndata received\n', data, '\n\n\n');
            
#             result = runDataPipelineBackend(data)
#             return result #{'results': eval}
#         except Exception as e:
#             print(e)
#             return str(e)

# api.add_resource(runDataPipeline, '/runDataPipeline')

# #get dataPipeline 
# class getDataPipeline(Resource):
#     def get(self, fileName):
#         return DB.getDataPipeline(fileName)

# api.add_resource(getDataPipeline, '/getDataPipeline/<string:fileName>')

# #list data pipelines
# class listDataPipelines(Resource):
#     def get(self):
#         return DB.listDataPipelines()

# api.add_resource(listDataPipelines, '/listDataPipelines')


# #save dataPipeline 
# saveDataPipeline_args = reqparse.RequestParser()
# saveDataPipeline_args.add_argument('dataPipelineData', type = str)
# class saveDataPipeline(Resource):
#     def post(self, fileName):
#         args = saveDataPipeline_args.parse_args()
#         DB.saveDataPipeline(fileName, json.loads(args.dataPipelineData))

# api.add_resource(saveDataPipeline, '/saveDataPipeline/<string:fileName>')

# #delete dataPipeline
# class deleteDataPipeline(Resource):
#     def post(self, fileName):
#         DB.deleteDataPipeline(fileName)

# api.add_resource(deleteDataPipeline, '/deleteDataPipeline/<string:fileName>')

#get model architecture
class getModelArchitecture(Resource):
    def get(self, fileName):
        return DB.getModelArchitecture(fileName)

api.add_resource(getModelArchitecture, '/getModelArchitecture/<string:fileName>')

# list model architecture
class listModelArchitectures(Resource):
    def get(self):
        return DB.listModelArchitectures()

api.add_resource(listModelArchitectures, '/listModelArchitectures')


#save model architecture
saveModelArchitecture_args = reqparse.RequestParser()
saveModelArchitecture_args.add_argument('modelData', type = str)
class saveModelArchitecture(Resource):
    def post(self, fileName):
        args = saveModelArchitecture_args.parse_args()
        DB.saveModelArchitecture(fileName, json.loads(args.modelData))

api.add_resource(saveModelArchitecture, '/saveModelArchitecture/<string:fileName>')

#delete model architecture
class deleteModelArchitecture(Resource):
    def post(self, fileName):
        DB.deleteModelArchitecture(fileName)

api.add_resource(deleteModelArchitecture, '/deleteModelArchitecture/<string:fileName>')


#get trained model info 
getTrainedModelinfo_args = reqparse.RequestParser()
getTrainedModelinfo_args.add_argument('fileName', type = str)
class getTrainedModelinfo(Resource):
    def get(self):
        args = getTrainedModelinfo_args.parse_args()
        return DB.getTrainedModelInfo(args.fileName)
        
api.add_resource(getTrainedModelinfo, '/getTrainedModelInfo')


class listTrainingHistory(Resource):
    def get(self):
        listOfTrainedModels = DB.listTrainedModel()
        # print(listOfTrainedModels)
        listOfInfo = []
        for trainedModel in listOfTrainedModels:
            listOfInfo.append(DB.getTrainedModelInfo(trainedModel))
            listOfInfo[-1]['name'] = trainedModel
        return listOfInfo
api.add_resource(listTrainingHistory, '/listTrainingHistory')

class deleteTrainedModel(Resource):
    def post(self, fileName):
        DB.deleteTrainedModel(fileName)
        return 'remove success'

api.add_resource(deleteTrainedModel, '/deleteTrainedModel/<string:fileName>')


# upload data Objects
uploadData_args = reqparse.RequestParser()
uploadData_args.add_argument('file', type=werkzeug.datastructures.FileStorage, location='files')   
uploadData_args.add_argument('name', type=str)   
class uploadData(Resource):
    def post(self):
        args = uploadData_args.parse_args()
        file = args['file']
        DB.saveDataObject(args['name'], file)
api.add_resource(uploadData, '/uploadData')

# list data Objects INFO (not datasets)
class listDataObjects(Resource):
    def get(self):
        listDataObjectsInfo = DB.listDataObjectsInfo()
        return listDataObjectsInfo
api.add_resource(listDataObjects, '/listDataObjects')

class deleteDataObjects(Resource):
    def post(self, fileName):
        DB.deleteDataObjects(fileName)
        return 'remove success'

api.add_resource(deleteDataObjects, '/deleteDataObjects/<string:fileName>')

# use model through api
useModel_args = reqparse.RequestParser()
useModel_args.add_argument('data', type = str)
class useModel(Resource):
    def post(self, modelName): 
        # print('called')
        args = useModel_args.parse_args()
        data = json.loads(args.data)
        model = DB.getTrainedModel(modelName)
        return model.predict(data)
api.add_resource(useModel, '/useModel/<string:modelName>')

# download model through api
exportModel_args = reqparse.RequestParser()
exportModel_args.add_argument('directory', type = str)
class exportModel(Resource):
    def post(self, modelName): 
        DB.exportTrainedModel(modelName, exportModel_args.parse_args().directory)
api.add_resource(exportModel, '/exportModel/<string:modelName>')




if __name__ == '__main__':
    app.run(debug=False, port=5000)
