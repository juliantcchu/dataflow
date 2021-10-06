from tensorflow import keras
import tensorflow.keras.layers as layers
# import numpy as np

class modelClass:

    def __init__(self, model_params):
        self.model = None
        self.layers = {}
        self.linked_layers = {}
        self.inputs = []
        self.outputs = []
        self.output_names = {}
        self.nodes, self.edges = model_params['nodes'], model_params['edges']
        for node in self.nodes:
            self.createLayer(node)


    def createLayer(self, layer_data):
        layer_type = layer_data['data']['module_name']
        params = layer_data['data']['params']
        Id = layer_data['id']
        if layer_type == 'Input':
            self.layers[Id] = layers.Input(name=params[0]['value'], shape=eval(params[1]['value']))#['shape'])
        elif layer_type == 'Output':
            self.layers[Id] = {'layer_type': 'dummy_output_layer', 'name': params[0]['value']}
        elif layer_type == 'Fully Connected Layer':
            self.layers[Id] = layers.Dense(eval(params[0]['value']), activation=params[1]['value'])
        elif layer_type == 'Concatenate':
            self.layers[Id] = layers.Concatenate(axis=eval(params[0]['value']))
        elif layer_type == 'Reshape':
            self.layers[Id] = layers.Reshape(eval(params[0]['value']))
        elif layer_type == '1D Convolutional Layer':
            self.layers[Id] = layers.Conv1D(filters=eval(params[0]['value']),kernel_size=eval(params[1]['value']),strides=eval(params[2]['value']), padding=params[3]['value'], activation=params[4]['value']) 
        elif layer_type == '2D Convolutional Layer':
            self.layers[Id] = layers.Conv2D(filters=eval(params[0]['value']),kernel_size=eval(params[1]['value']),strides=eval(params[2]['value']), padding=params[3]['value'], activation=params[4]['value']) 
        elif layer_type == '3D Convolutional Layer':
            self.layers[Id] = layers.Conv3D(filters=eval(params[0]['value']),kernel_size=eval(params[1]['value']),strides=eval(params[2]['value']), padding=params[3]['value'], activation=params[4]['value']) 
        elif layer_type == '1D Maxpooling Layer':
            self.layers[Id] = layers.MaxPool1D(pool_size=eval(params[0]['value']), strides=eval(params[1]['value']))
        elif layer_type == '2D Maxpooling Layer':
            self.layers[Id] = layers.MaxPool2D(pool_size=eval(params[0]['value']), strides=eval(params[1]['value']))
        elif layer_type == '3D Maxpooling Layer':
            self.layers[Id] = layers.MaxPool3D(pool_size=eval(params[0]['value']), strides=eval(params[1]['value'])) 
        elif layer_type == 'Long short-term memory (LSTM)':
            self.layers[Id] = layers.LSTM(eval(params[0]['value']), activation=params[1]['value'], use_bias=params[2]['value'],return_sequences=params[3]['value'])
        elif layer_type == 'Dropout':
            self.layers[Id] = layers.Dropout(eval(params[0]['value']))
        elif layer_type == 'Flatten':
            self.layers[Id] = layers.Flatten()

        #edit
        
    def checkExist(self, nodeIds):
        if type(nodeIds) != list:
            nodeIds = [nodeIds]
        for nodeId in nodeIds:
            if nodeId not in self.linked_layers:
                return False
        return True

    def getPrevNodeIds(self, nodeId):
        prereqs = []
        for edge in self.edges:
            if edge['target'] == nodeId:
                prereqs.append(edge['source'])
        return prereqs

    def getNextNodeIds(self, nodeId):
        nextNodes = []
        for edge in self.edges:
            if edge['source'] == nodeId:
                nextNodes.append(edge['target'])
        return nextNodes
    



    def createModelInteriorHelper(self, currNodeId):

        prevNodeIds = self.getPrevNodeIds(currNodeId)
        # stop if not all prereqs are fulfilled
        if self.checkExist(prevNodeIds) == False:
            return
        
        if len(prevNodeIds) > 0:
            prevLayers = []
            for prevNodeId in prevNodeIds:
                prevLayers.append(self.linked_layers[prevNodeId])
            if len(prevLayers) == 1:
                prevLayers = prevLayers[0]
            self.linked_layers[currNodeId] = self.layers[currNodeId] (prevLayers) #check for errors
        
        nextNodeIds = self.getNextNodeIds(currNodeId)
        for nextNodeId in nextNodeIds:
            if type(self.layers[nextNodeId]) is dict and self.layers[nextNodeId]['layer_type']  == 'dummy_output_layer':
                self.outputs.append(self.linked_layers[currNodeId])
                self.output_names[self.layers[nextNodeId]['name']] = self.layers[currNodeId].name
            else:
                self.createModelInteriorHelper(nextNodeId)



    def createModel(self, print_summary = False):
        
        #find input nodes
        inputNodeIds = [node['id'] for node in self.nodes if node['data']['module_name'] == 'Input']
        for inputNodeId in inputNodeIds:
            self.linked_layers[inputNodeId] = self.layers[inputNodeId]
            self.inputs.append(self.linked_layers[inputNodeId])
            nextNodes = self.getNextNodeIds(inputNodeId)
            for nextNode in nextNodes:
                self.createModelInteriorHelper(nextNode)
        
        self.model = keras.Model(inputs=self.inputs, outputs=self.outputs)
        if print_summary:
            print(self.model.summary())
        return self.model
    
    def getOutputNames(self):
        return self.output_names
    
        



    
