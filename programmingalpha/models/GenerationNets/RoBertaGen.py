import torch
from onmt.encoders.transformer import EncoderBase
from fairseq.modules import TransformerSentenceEncoder
import os
from typing import Optional, Tuple
from torch.nn import functional as F
from programmingalpha.models import expandEmbeddingByN

class OnmtRobertaEncoder(EncoderBase):
    '''
    Returns:
        (torch.FloatTensor, torch.FloatTensor):

        * embeddings ``(src_len, batch_size, model_dim)``
        * memory_bank ``(src_len, batch_size, model_dim)``
    '''

    def __init__(self, model_path, padding_idx, vocab_size):
        super(OnmtRobertaEncoder, self).__init__()
        

        self.roberta_encoder = TransformerSentenceEncoder(
            padding_idx=padding_idx,
            vocab_size=vocab_size,
            num_encoder_layers=args.encoder_layers,
            embedding_dim=args.encoder_embed_dim,
            ffn_embedding_dim=args.encoder_ffn_embed_dim,
            num_attention_heads=args.encoder_attention_heads,
            dropout=args.dropout,
            attention_dropout=args.attention_dropout,
            activation_dropout=args.activation_dropout,
            max_seq_len=args.max_positions,
            num_segments=0,
            encoder_normalize_before=True,
            apply_bert_init=True,
            activation_fn=args.activation_fn,
        )
        print(self.roberta_encoder)
        print("defined the roberta network!")
        model_ckpt_file=os.path.join(model_path, "model.pt")
        if os.path.exists(model_ckpt_file):
            ckpt = torch.load(model_ckpt_file, map_location='cpu')
            args = ckpt["args"]
            model_dict = {}
            for k, v in ckpt["model"].items():
                if "decoder.sentence_encoder." in k:
                    k = k.replace("decoder.sentence_encoder.", "")
                    if k not in self.roberta_encoder.state_dict().keys():
                        print("skip", k)
                        continue
                    model_dict[k] = v
                    print("{}:{}".format(k, v.size()))

            self.roberta_encoder.load_state_dict(model_dict)
            print("loaded {}/{} weights".format(len(model_dict.keys()), len(self.roberta_encoder.state_dict().keys())))

        self.roberta_encoder.embed_tokens=expandEmbeddingByN(self.roberta_encoder.embed_tokens, 4 )
        print("*"*50)


    def forward(self, src, lengths=None):
        """See :func:`EncoderBase.forward()`"""
        self._check_args(src, lengths)
        src=src.squeeze(2).transpose(0,1).contiguous()

        #outs, sent_out=self.roberta_encoder(src)
        emb, outs, sent_out=self.forwad1(self.roberta_encoder,src)

        #emb=outs[0]

        out=outs[-1]
        #print("src--> outs", src.size(), out.size(), emb.size())
        #return emb.transpose(0,1).contiguous(), out.transpose(0, 1).contiguous(), lengths
        return emb, out, lengths

def getWordEmbeddingFromRoberta(model:OnmtRobertaEncoder):
    return model.roberta_encoder.embed_tokens

def buildRoberta(**kwargs):

    if "model_path" not in kwargs:
        import programmingalpha
        kwargs["model_path"] = programmingalpha.RoBertaBase
    if "padding_idx" not in kwargs:
        kwargs["padding_idx"] = 1
    if "vocab_size" not in kwargs:
        kwargs["vocab_size"] =50265

    encoder=OnmtRobertaEncoder(kwargs["model_path"], kwargs["padding_idx"], kwargs["vocab_size"])
    return encoder
