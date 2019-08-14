from .BertGen import buildBert
from .RoBertaGen import buildRoberta
from .XLNetGen import buildXLNet

supported_encoder_builders={
    "bert":buildBert,
    "roberta":buildRoberta,
    "xlnet":buildXLNet
}

