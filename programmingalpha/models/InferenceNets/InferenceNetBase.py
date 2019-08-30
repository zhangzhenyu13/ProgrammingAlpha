from pytorch_transformers import PreTrainedModel, PretrainedConfig
from torch import nn
class InferenceNet(nn.Module):
    def _load_model(self, encoder_cls:PreTrainedModel, config_cls:PretrainedConfig):
            if self.model_path is  None or not self.model_path:
                    config=config_cls()
                    encoder=encoder_cls(config)
                    return encoder, config

            return encoder_cls.from_pretrained(self.model_path),  config_cls.from_pretrained(self.model_path)

    def __init__(self, model_path=None, num_labels=4):
            nn.Module.__init__(self)
            self.model_path=model_path
            self.num_labels=num_labels

            self.encoder=None
            self.config=None

    def forward(self, input_ids, token_type_ids=None, labels=None):
        raise NotImplementedError