import json
import os

_ALphaRootPath="/home/zhangzy/"
class AlphaPathLookUp(object):
    #bert model
    BertRoot=os.path.join(_ALphaRootPath,"ShareModels/bert/")
    BertBaseCased=os.path.join(_ALphaRootPath,"ShareModels/bert/base-cased/")
    BertBaseUnCased=os.path.join(_ALphaRootPath,"ShareModels/bert/base-uncased/")
    BertLargeCased=os.path.join(_ALphaRootPath,"ShareModels/bert/large-cased/")
    BertLargeUnCased=os.path.join(_ALphaRootPath,"ShareModels/bert/large-uncased/")
    BertLargeCasedMasking=os.path.join(_ALphaRootPath,"ShareModels/bert/large-cased-whole-masking/")
    BertLargeUnCasedMasking=os.path.join(_ALphaRootPath,"ShareModels/bert/large-uncased-whole-masking/")
    #gpt-2 model
    GPT2Base=os.path.join(_ALphaRootPath,"ShareModels/gpt-2/base/")
    GPT2Medium=os.path.join(_ALphaRootPath,"ShareModels/gpt-2/medium/")
    #XLNet
    XLNetBaseCased=os.path.join(_ALphaRootPath,"ShareModels/xlnet/base-cased")
    XLNetLargeCased=os.path.join(_ALphaRootPath,"ShareModels/xlnet/large-cased")
    #RoBerta
    RoBertaBase=os.path.join(_ALphaRootPath,"ShareModels/roberta/base/")
    RoBertaLarge=os.path.join(_ALphaRootPath,"ShareModels/roberta/large/")

    #global project path
    ConfigPath=os.path.join(_ALphaRootPath,"ProgrammingAlpha/ConfigData/")
    DataCases=os.path.join(_ALphaRootPath,"ProgrammingAlpha/dataCases/")

    DataPath=os.path.join(_ALphaRootPath,"ProjectData/")
    ModelPath=os.path.join(_ALphaRootPath,"ProjectModels/")

    def update_from_file(self,filename):
        with open(filename, "r", encoding="utf-8") as f:
            paths=json.load(f)
        for k,v in paths.items():
            self.__setattr__(k,v)
    
class AlphaConfig(object):
    def __init__(self, config_dict:dict):
        for k, v in config_dict.items():
            self.__setattr__(k,v)
    
    @classmethod
    def loadConfig(cls, filename):
        with open(filename,"r") as f:
            config=json.load(f)
        config=cls(config)
        return config

    def saveConfig(self, config, filename):
        with open(filename, "w") as f:
            if type(config)==AlphaConfig:
                config=config.__dict__
            json.dump(config,f)
