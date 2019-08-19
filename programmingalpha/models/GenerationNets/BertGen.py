from pytorch_transformers.modeling_bert import BertModel, BertConfig
from onmt.modules.embeddings import Embeddings
from onmt.encoders.transformer import EncoderBase
import os
from programmingalpha.models import expandEmbeddingByN

class OnmtBertEncoder(EncoderBase):
    def __init__(self, model_path):
        super(OnmtBertEncoder, self).__init__()
        config=BertConfig.from_json_file(os.path.join( model_path, "config.json") )
        model=BertModel.from_pretrained(pretrained_model_name_or_path=os.path.join( model_path, "pytorch_model.bin"), config=config)
        model.embeddings.word_embeddings=expandEmbeddingByN(model.embeddings.word_embeddings, 4)
        model.embeddings.word_embeddings=expandEmbeddingByN(model.embeddings.word_embeddings, 2, last=True)

        self.encoder=model
        print("init BERT model with {} weights".format(len(self.encoder.state_dict())))
        #print(model)
        print("***"*20)

        
    def forward(self, src, lengths=None):
        """
        Args:
            src (LongTensor):
               padded sequences of sparse indices ``(src_len, batch, nfeat)``
            lengths (LongTensor): length of each sequence ``(batch,)``

        """
        #print("input->", src.size())
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

def getWordEmbeddingFromBert(model:OnmtBertEncoder):
    return model.encoder.embeddings.word_embeddings

def buildBert(**kwargs):
    if "model_path" not in kwargs:
        import programmingalpha
        kwargs["model_path"] = programmingalpha.BertBaseUnCased

    encoder=OnmtBertEncoder(kwargs["model_path"])

    return encoder