from torch import nn


class HuggingFaceModel(nn.Module):
    def __init__(self, model_class, model_name, num_channels, num_labels):
        super().__init__()
        model = model_class.from_pretrained(
            model_name, num_labels=num_labels, ignore_mismatched_sizes=True
        )
        config = model.config
        config.backbone_config.num_channels = num_channels
        self.model = model_class.from_pretrained(
            model_name, config=config, ignore_mismatched_sizes=True
        )

    def forward(self, data):
        return self.model(data)[0]
