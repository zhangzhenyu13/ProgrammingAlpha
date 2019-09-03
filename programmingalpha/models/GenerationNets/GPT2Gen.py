from pytorch_transformers import GPT2Config, GPT2Model
from onmt.modules.embeddings import Embeddings
from onmt.encoders.transformer import EncoderBase
import os
from programmingalpha.models import expandEmbeddingByN

class OnmtGPT2Encoder(EncoderBase):
    def __init__(self, model_path):
        super(OnmtGPT2Encoder, self).__init__()
        config=GPT2Config.from_json_file(os.path.join( model_path, "config.json") )
        pretrained_dict=os.path.join( model_path, "pytorch_model.bin")
        if os.path.exists(pretrained_dict):
            model=GPT2Model.from_pretrained(pretrained_model_name_or_path=pretrained_dict, config=config)
            print("init GPT2 model with {} weights".format(len(model.state_dict())))
        else:
            model=GPT2Model(config)

        model.wte=expandEmbeddingByN(model.wte, 4)
        self.encoder=model

        #print(model)
        print("***"*20)

        
    def forward(self, src, lengths=None):
        """
        Args:
            src (LongTensor):
               padded sequences of sparse indices ``(src_len, batch, nfeat)``
            lengths (LongTensor): length of each sequence ``(batch,)``

        """
        inputids=src.squeeze(2).transpose(0,1).contiguous()

        outputs=self.encoder(input_ids=inputids)
        #print(len(outputs))
        #print(outputs)

        emb=outputs[2][-1]
        memory_bank=outputs[0]

        emb=emb.transpose(0,1).contiguous()
        memory_bank=memory_bank.transpose(0,1).contiguous()

        #print("src--> outs", src.size(), emb.size(), memory_bank.size())

        return emb, memory_bank, lengths    

def getWordEmbeddingFromGPT2Encoder(model:OnmtGPT2Encoder):
    return model.encoder.wte

def buildGPT2(**kwargs):
    if "model_path" not in kwargs:
        import programmingalpha
        kwargs["model_path"] = programmingalpha.GPT2Base

    encoder=OnmtGPT2Encoder(kwargs["model_path"])

    return encoder