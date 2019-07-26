import torch
import torch.nn as nn
import onmt.utils
from pytorch_transformers.modeling_xlnet import XLNetModel
import programmingalpha
import logging
import numpy as np
import math

from copy import deepcopy

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)



class OnmtXLNetEncoder(XLNetModel):

    def setEmbeddings(self, emb):
        self.embeddings = emb

    def forward(self, input_ids, token_type_ids=None, input_mask=None, attention_mask=None,
                mems=None, perm_mask=None, target_mapping=None, head_mask=None):
        # the original code for XLNet uses shapes [len, bsz] with the batch dimension at the end
        # but we want a unified interface in the library with the batch size on the first dimension
        # so we move here the first dimension (batch) to the end
        input_ids = input_ids.transpose(0, 1).contiguous()
        token_type_ids = token_type_ids.transpose(0, 1).contiguous() if token_type_ids is not None else None
        input_mask = input_mask.transpose(0, 1).contiguous() if input_mask is not None else None
        attention_mask = attention_mask.transpose(0, 1).contiguous() if attention_mask is not None else None
        perm_mask = perm_mask.permute(1, 2, 0).contiguous() if perm_mask is not None else None
        target_mapping = target_mapping.permute(1, 2, 0).contiguous() if target_mapping is not None else None

        qlen, bsz = input_ids.shape[0], input_ids.shape[1]
        mlen = mems[0].shape[0] if mems is not None else 0
        klen = mlen + qlen

        dtype_float = next(self.parameters()).dtype
        device = next(self.parameters()).device

        ##### Attention mask
        # causal attention mask
        if self.attn_type == 'uni':
            attn_mask = self.create_mask(qlen, mlen)
            attn_mask = attn_mask[:, :, None, None]
        elif self.attn_type == 'bi':
            attn_mask = None
        else:
            raise ValueError('Unsupported attention type: {}'.format(self.attn_type))

        # data mask: input mask & perm mask
        assert input_mask is None or attention_mask is None, "You can only use one of input_mask (uses 1 for padding) "
        "or attention_mask (uses 0 for padding, added for compatbility with BERT). Please choose one."
        if input_mask is None and attention_mask is not None:
            input_mask = 1.0 - attention_mask
        if input_mask is not None and perm_mask is not None:
            data_mask = input_mask[None] + perm_mask
        elif input_mask is not None and perm_mask is None:
            data_mask = input_mask[None]
        elif input_mask is None and perm_mask is not None:
            data_mask = perm_mask
        else:
            data_mask = None

        if data_mask is not None:
            # all mems can be attended to
            mems_mask = torch.zeros([data_mask.shape[0], mlen, bsz]).to(data_mask)
            data_mask = torch.cat([mems_mask, data_mask], dim=1)
            if attn_mask is None:
                attn_mask = data_mask[:, :, :, None]
            else:
                attn_mask += data_mask[:, :, :, None]

        if attn_mask is not None:
            attn_mask = (attn_mask > 0).to(dtype_float)

        if attn_mask is not None:
            non_tgt_mask = -torch.eye(qlen).to(attn_mask)
            non_tgt_mask = torch.cat([torch.zeros([qlen, mlen]).to(attn_mask), non_tgt_mask], dim=-1)
            non_tgt_mask = ((attn_mask + non_tgt_mask[:, :, None, None]) > 0).to(attn_mask)
        else:
            non_tgt_mask = None

        ##### Word embeddings and prepare h & g hidden states
        word_emb_k = self.word_embedding(input_ids)
        output_h = self.dropout(word_emb_k)
        if target_mapping is not None:
            word_emb_q = self.mask_emb.expand(target_mapping.shape[0], bsz, -1)
        # else:  # We removed the inp_q input which was same as target mapping
        #     inp_q_ext = inp_q[:, :, None]
        #     word_emb_q = inp_q_ext * self.mask_emb + (1 - inp_q_ext) * word_emb_k
            output_g = self.dropout(word_emb_q)
        else:
            output_g = None

        ##### Segment embedding
        if token_type_ids is not None:
            # Convert `token_type_ids` to one-hot `seg_mat`
            mem_pad = torch.zeros([mlen, bsz], dtype=torch.long, device=device)
            cat_ids = torch.cat([mem_pad, token_type_ids], dim=0)

            # `1` indicates not in the same segment [qlen x klen x bsz]
            seg_mat = (token_type_ids[:, None] != cat_ids[None, :]).long()
            seg_mat = F.one_hot(seg_mat, num_classes=2).to(dtype_float)
        else:
            seg_mat = None

        ##### Positional encoding
        pos_emb = self.relative_positional_encoding(qlen, klen, bsz=bsz)
        pos_emb = self.dropout(pos_emb)

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads] (a head_mask for each layer)
        # and head_mask is converted to shape [num_hidden_layers x qlen x klen x bsz x n_head]
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)
                head_mask = head_mask.expand(self.n_layer, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(1).unsqueeze(1)
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype) # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.n_layer

        new_mems = ()
        if mems is None:
            mems = [None] * len(self.layer)

        attentions = []
        hidden_states = []
        for i, layer_module in enumerate(self.layer):
            # cache new mems
            new_mems = new_mems + (self.cache_mem(output_h, mems[i]),)
            if self.output_hidden_states:
                hidden_states.append((output_h, output_g) if output_g is not None else output_h)

            outputs = layer_module(output_h, output_g, attn_mask_h=non_tgt_mask, attn_mask_g=attn_mask,
                                   r=pos_emb, seg_mat=seg_mat, mems=mems[i], target_mapping=target_mapping,
                                   head_mask=head_mask[i])
            output_h, output_g = outputs[:2]
            if self.output_attentions:
                attentions.append(outputs[2])

        # Add last hidden state
        if self.output_hidden_states:
            hidden_states.append((output_h, output_g) if output_g is not None else output_h)

        output = self.dropout(output_g if output_g is not None else output_h)

        # Prepare outputs, we transpose back here to shape [bsz, len, hidden_dim] (cf. beginning of forward() method)
        outputs = (output.permute(1, 0, 2).contiguous(), new_mems)
        if self.output_hidden_states:
            if output_g is not None:
                hidden_states = tuple(h.permute(1, 0, 2).contiguous() for hs in hidden_states for h in hs)
            else:
                hidden_states = tuple(hs.permute(1, 0, 2).contiguous() for hs in hidden_states)
            outputs = outputs + (hidden_states,)
        if self.output_attentions:
            attentions = tuple(t.permute(2, 3, 0, 1).contiguous() for t in attentions)
            outputs = outputs + (attentions,)

        return outputs  # outputs, new_mems, (hidden_states), (attentions)



class TransformerModel(nn.Module):
    def __init__(self, encoder: OnmtBertTransformerEncoder, decoder: onmt.decoders.TransformerDecoder):
        nn.Module.__init__(self)

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt, lengths, bptt=False):
        # print("lengths, src and tgt size is ",lengths.size(),src.size(),tgt.size())
        # src_enc=src.squeeze(2)

        tgt = tgt[:-1]  # exclude last target from inputs

        enc_state, memory_bank, lengths = self.encoder(input_ids=src, lengths=lengths)
        # print("size of encoded layers",memory_bank.size())

        if bptt is False:
            self.decoder.init_state(src, memory_bank, enc_state)
        dec_out, attns = self.decoder(tgt, memory_bank,
                                      memory_lengths=lengths)
        # print("decoded result",dec_out.size(),attns["std"].size())
        return dec_out, attns


class TextGeneratorModel(object):
    opt = None
    model_opt = None
    fields = None
    gpu_id = None
    device = None

    def __init__(self):
        self.build_model()

    def build_generator(self):
        model_opt = self.model_opt
        fields = self.fields

        from onmt.modules.util_class import Cast

        # Build Generator.
        if not model_opt.copy_attn:

            if model_opt.generator_function == "sparsemax":
                gen_func = onmt.modules.sparse_activations.LogSparsemax(dim=-1)
            else:
                gen_func = nn.LogSoftmax(dim=-1)
            generator = nn.Sequential(
                nn.Linear(model_opt.dec_rnn_size,
                          len(fields["tgt"].base_field.vocab)),
                Cast(torch.float32),
                gen_func
            )
            if model_opt.share_decoder_embeddings:
                generator[0].weight = self.transformer.decoder.embeddings.word_embeddings.weight
        else:
            tgt_base_field = fields["tgt"].base_field
            vocab_size = len(tgt_base_field.vocab)
            pad_idx = tgt_base_field.vocab.stoi[tgt_base_field.pad_token]
            generator = onmt.modules.CopyGenerator(model_opt.dec_rnn_size, vocab_size, pad_idx)

        return generator

    def use_device(self):
        from onmt.utils.misc import use_gpu
        from torch import cuda
        gpu = use_gpu(self.opt) and cuda.is_available()
        gpu_id = self.gpu_id

        if gpu and gpu_id is not None:
            device = torch.device("cuda", gpu_id)
        elif gpu and not gpu_id:
            device = torch.device("cuda")
        elif not gpu:
            device = torch.device("cpu")

        self.device = device
        logger.info("gpu: {}, gpu_id: {}, device: {}".format(gpu, gpu_id, device))

        self.transformer.to(device)
        if self.model_opt.model_dtype == 'fp16':
            self.transformer.half()

        logger.info("device:{},half:{}".format(device, self.model_opt.model_dtype))

    def build_model(self):

        # encoder
        bert = OnmtBertTransformerEncoder.from_pretrained(programmingalpha.BertBasePath)

        def __copyEmbeddings(embeddings: nn.Embedding, index1, index2):
            # print(embeddings.weight.size())
            weight = embeddings.weight.detach().numpy()

            weight[index2] = deepcopy(weight[index1])

            weight = torch.tensor(weight, requires_grad=True)
            embeddings.weight = nn.Parameter(weight, requires_grad=True)

        __copyEmbeddings(bert.embeddings.word_embeddings, 0, 1)
        __copyEmbeddings(bert.embeddings.word_embeddings, 100, 0)

        tgt_base_field = self.fields["tgt"].base_field
        vocab_size = len(tgt_base_field.vocab)
        pad_idx = tgt_base_field.vocab.stoi[tgt_base_field.pad_token]
        model_dim = self.model_opt.rnn_size  # if hasattr(self.model_opt,"rnn_size") else 768
        max_seq_length = 5000
        max_relative_position = 512
        heads = 12
        feed_forwad_size = 3072
        drop_rate = self.model_opt.dropout  # if hasattr(self.model_opt,"dropout") else 0
        layers = self.model_opt.layers  # if hasattr(self.model_opt,"layers") else 4
        # tgt embeddings

        transformerEmb = OnmtBertEmbedding(vocab_size, model_dim, max_seq_length, drop_rate, pad_idx, bert.embeddings)

        bert.setEmbeddings(transformerEmb)
        # transformerEmb=BertAsEmbedding(bert,self.tgt_padding)

        # decoder
        transformerDecoder = onmt.decoders.TransformerDecoder(
            num_layers=layers, d_model=model_dim, heads=heads, d_ff=feed_forwad_size,
            copy_attn=True, self_attn_type="scaled-dot", dropout=drop_rate, embeddings=transformerEmb,
            max_relative_positions=max_relative_position
        )

        self.transformer = TransformerModel(bert, transformerDecoder)

        self.transformer.generator = self.build_generator()

        self.use_device()

    def loadModel(self, model_file=None, checkpoint=None):
        assert model_file is not None or checkpoint is not None

        if checkpoint is None:
            # logger.info("loading from model file")
            model_dict = torch.load(model_file)
        else:
            # logger.info("loading from model checkpoint")
            model_dict = checkpoint

        weight_dict = model_dict["model"]
        generator_dict = model_dict["generator"]
        # print(weight_dict.keys())
        # print(generator_dict.keys())
        for k in generator_dict:
            weight_dict["generator." + k] = generator_dict[k]

        transformer_dict = self.transformer.state_dict()
        exlude_dict = (
        "encoder.embeddings.token_type_embeddings.weight", "encoder.embeddings.position_embeddings.weight",
        "decoder.embeddings.position_embeddings.weight")
        restore_dict = {}

        layers = self.model_opt.layers  # if hasattr(self.model_opt,"layers") else 4
        logger.info("decoder layer num: %d" % layers)

        # self.transformer.load_state_dict(weight_dict)
        for k, v in weight_dict.items():
            if k in exlude_dict:
                logger.info("skip :{}".format(k))
                continue
            restore_dict[k] = v
        transformer_dict.update(restore_dict)

        if model_file:
            logger.info("init model weight with " + model_file)
        else:
            logger.info("init model with checkpoint")
