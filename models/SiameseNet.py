import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18 ,ResNet18_Weights
import models.ResNet as ResNet

# -------------------------------------------------
# 1. 获取去掉最后FC层的 ResNet18 作为骨干网络
# -------------------------------------------------
def get_resnet18_backbone(customize_network = None ,pretrained: bool = False) -> nn.Module:
    """
    返回一个去掉了最后线性层(FC层)的 ResNet18，用于提取特征。
    """
    if customize_network is not None:
        return customize_network(pretrained=pretrained)
    else:
        # 使用 torchvision 内置的 resnet18
        if pretrained:
            backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        else:
            backbone = resnet18()
        # 将最后的全连接层替换为 Identity，
        # 这样 forward 的输出就是 global avg pool 之后的特征向量
        backbone.fc = nn.Identity()
        return backbone

# -------------------------------------------------
# 2. 定义 Siamese Network (孪生网络)
# -------------------------------------------------
class SiameseNetwork(nn.Module):
    def __init__(self, customize_network = None,pretrained: bool = False, **kwargs):
        """
        Args:
            customize_network: 自定义网络类
            pretrained (bool): 是否使用预训练权重
            **kwargs: 可选参数
                - feature_dim (int): 特征维度 (默认: 3)
                - layer_width (list<int>): 隐藏层宽度 (默认: [256, 128, 64])
        """
        super(SiameseNetwork, self).__init__()
        
        # 共享骨干网络（去掉FC层的ResNet18），只会在内存中存在一份权重
        self.base_model = get_resnet18_backbone(customize_network=customize_network,pretrained=pretrained)
        
        # 如果需要额外的层，可以加在这里
        # 例如：将 512 维的特征映射到更小(或更大)维度
        feature_dim = kwargs.get('feature_dim', 3)
        layer_width = kwargs.get('layer_width', [512, 256, 64])
        embedding_width = kwargs.get('embedding_width',[64])
        using_embedding = kwargs.get('using_embedding',False)
        layer_width.append(feature_dim)
        extra_layer_input = 512*2
        self.embedding_layer = None
        if using_embedding:
            extra_layer_input = (512 + embedding_width[-1])*2
            self.embedding_layer = self.make_fc_layers(4,embedding_width)
            self.extra_layer = self.make_fc_layers(extra_layer_input, layer_width)
        else:
            self.extra_layer = self.make_fc_layers(extra_layer_input, layer_width)

    def make_fc_layers(self, in_features: int, layer_width: list) -> nn.Module:
        """
        构建全连接层
        Args:
            in_features: 输入特征维度
            layer_width: 各隐藏层的宽度
        """
        layers = []
        for width in layer_width:
            layers.append(nn.Linear(in_features, width))
            layers.append(nn.ReLU(inplace=True))
            in_features = width
        return nn.Sequential(*layers)
    
    def forward_once(self, x: torch.Tensor,embedding_info: torch.Tensor = None) -> torch.Tensor:
        """
        只负责执行一次 forward，用于处理单张图片。
        """
        # 从 ResNet18 骨干中获取 512 维特征
        features = self.base_model(x)
        # 加入额外的 embedding 层，编码xyxy信息
        if embedding_info is not None:
            embedding_feat = self.embedding_layer(embedding_info)
            features = torch.cat((features,embedding_feat),dim=1)
    
        return features

    def forward(self, x1: torch.Tensor, x2: torch.Tensor,\
                embedding_info1: torch.Tensor = None,embedding_info2: torch.Tensor = None
                ) -> torch.Tensor:
        """
        同时处理两张输入图片，将输出特征向量进行拼接后返回。
        """
        feat1 = self.forward_once(x1,embedding_info1)  # [batch_size, 512]
        feat2 = self.forward_once(x2,embedding_info2)  # [batch_size, 512]
        
        # 拼接特征向量，拼接后的 shape: [batch_size, 512*2]
        out = torch.cat((feat1, feat2), dim=1)
        
        # 如果有额外层，进一步处理
        out = self.extra_layer(out)
        
        return out


# -------------------------------------------------
# 3. 测试示例
# -------------------------------------------------
if __name__ == "__main__":
    # 设备选择：如果有GPU可用，则使用GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建 Siamese 网络实例，选择是否加载预训练权重
    customize_network = ResNet.resnet18
    siamese_model = SiameseNetwork(customize_network=customize_network,pretrained=True).to(device)
    
    # 假设我们有两批图片，每批大小为 (batch_size, 3, 224, 224)
    # 这里使用随机张量模拟，实际中应换成真实图像经过 transforms 后的张量
    batch_size = 4
    img1 = torch.randn(batch_size, 3, 224, 224).to(device)
    img2 = torch.randn(batch_size, 3, 224, 224).to(device)
    
    # 前向传播
    output = siamese_model(img1, img2)
    
    # 打印结果
    print("输出张量维度:", output.shape)  # [batch_size, 1024] = [4, 1024]
    print("输出示例:", output)
   
    
    from utils.VisualizeParam import visualize_conv_filters
    # for name, module in siamese_model.named_modules(): print(name, module)
    visualize_conv_filters(siamese_model.base_model, layer_name='conv1', max_filters=8)