import json
import os
#bert model
BertRoot="/home/LAB/zhangzy/ShareModels/bert/"
BertBaseCased="/home/LAB/zhangzy/ShareModels/bert/base-cased/"
BertBaseUnCased="/home/LAB/zhangzy/ShareModels/bert/base-uncased/"
BertLargeCased="/home/LAB/zhangzy/ShareModels/bert/large-cased/"
BertLargeUnCased="/home/LAB/zhangzy/ShareModels/bert/large-uncased/"
BertLargeCasedMasking="/home/LAB/zhangzy/ShareModels/bert/large-cased-whole-masking/"
BertLargeUnCasedMasking="/home/LAB/zhangzy/ShareModels/bert/large-uncased-whole-masking/"
#gpt-2 model
GPT2Base="/home/LAB/zhangzy/ShareModels/gpt-2/base/"
GPT2Medium="/home/LAB/zhangzy/ShareModels/gpt-2/medium/"
#XLNet
XLNetBaseCased="/home/LAB/zhangzy/ShareModels/xlnet/base-cased"
XLNetLargeCased="/home/LAB/zhangzy/ShareModels/xlnet/large-cased"
#RoBerta
RoBertaBase="/home/LAB/zhangzy/ShareModels/roberta/base/"
RoBertaLarge="/home/LAB/zhangzy/ShareModels/roberta/large/"

#global project path
ConfigPath="/home/LAB/zhangzy/ProgrammingAlpha/ConfigData/"
DataCases="/home/LAB/zhangzy/ProgrammingAlpha/dataCases/"

DataPath="/home/LAB/zhangzy/ProjectData/"
ModelPath="/home/LAB/zhangzy/ProjectModels/"



class AlphaConfig(object):
    def __init__(self, config_dict:dict):
        for k, v in config_dict.items():
            self.__setattr__(k,v)

def loadConfig(filename):
    with open(filename,"r") as f:
        config=json.load(f)
    config=AlphaConfig(config)
    return config

def saveConfig(filename,config):
    with open(filename, "w") as f:
        if type(config)==AlphaConfig:
            config=config.__dict__
        json.dump(config,f)


