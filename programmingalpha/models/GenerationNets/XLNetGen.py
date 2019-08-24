from pytorch_transformers import XLNetModel, XLNetConfig
from onmt.encoders.transformer import EncoderBase
import os
from programmingalpha.models import expandEmbeddingByN

class OnmtXLNetEncoder(EncoderBase):
    '''
        Returns:
            (torch.FloatTensor, torch.FloatTensor):

            * embeddings ``(src_len, batch_size, model_dim)``
            * memory_bank ``(src_len, batch_size, model_dim)``
        '''
    def __init__(self, model_path):
        super(OnmtXLNetEncoder, self).__init__()
        config=XLNetConfig.from_json_file(os.path.join( model_path, "config.json") )
        model=XLNetModel.from_pretrained(pretrained_model_name_or_path= os.path.join( model_path, "pytorch_model.bin"), config=config)
        model.word_embedding=expandEmbeddingByN(model.word_embedding, 4)
        model.word_embedding=expandEmbeddingByN(model.word_embedding, 2, last=True)
        self.encoder=model
        print("init XLNet model with {} weights".format(len(self.encoder.state_dict())))
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

def getWordEmbeddingFromXLNetEncoder(model:OnmtXLNetEncoder):
    return model.encoder.word_embedding

def buildXLNet(**kwargs):
    if "model_path" not in kwargs:
        import programmingalpha
        kwargs["model_path"] = programmingalpha.XLNetBaseCased

    encoder=OnmtXLNetEncoder(kwargs["model_path"])

    return encoder
