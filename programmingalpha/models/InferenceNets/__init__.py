from .import BertInferenceNet, XLNetInferenceNet

num_labels=4

def get_inference_net(model_path, name="bert"):
        
        model=None
        if name=="bert":
                model=BertInferenceNet.LinkNet(model_path, num_labels)
        if name=="bert_attn":
                model=BertInferenceNet.KnowNet(model_path, num_labels)
        if name=="xlnet":
                model =XLNetInferenceNet.LinkNet(model_path, num_labels)
        
        if model is None:
                raise ValueError("model {} is not supported yet!".format(name) )
        
        return model

def construct_inference_net(name="bert"):
        model=None
        if name=="bert":
                model=BertInferenceNet.LinkNet( num_labels=num_labels)
        if name=="bert_attn":
                model=BertInferenceNet.KnowNet(num_labels=num_labels)
        if name=="xlnet":
                model =XLNetInferenceNet.LinkNet( num_labels=num_labels)
        
        if model is None:
                raise ValueError("model {} is not supported yet!".format(name) )
        
        return model