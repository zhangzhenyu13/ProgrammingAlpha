from .BertGen import buildBert, getWordEmbeddingFromBert
from .RoBertaGen import buildRoberta, getWordEmbeddingFromRoberta
from .XLNetGen import buildXLNet, getWordEmbeddingFromXLNetEncoder
from .GPT2Gen import buildGPT2, getWordEmbeddingFromGPT2Encoder
supported_encoder_builders={
    "bert":buildBert,
    "roberta":buildRoberta,
    "xlnet":buildXLNet,
    "gpt2":buildGPT2
}

supported_embedding_extractors={
    "bert":getWordEmbeddingFromBert,
    "roberta":getWordEmbeddingFromRoberta,
    "xlnet":getWordEmbeddingFromXLNetEncoder,
    "gpt2":getWordEmbeddingFromGPT2Encoder
}