import torch
import torchvision

class FPN(torch.nn.Module):
    def __init__(self):
        super(FPN).__init__()

    def forward(self):
        pass


class RPN(torch.nn.Module):
    def __init__(self):
        super(RPN).__init__()

    def forward(self):
        pass


class ROIPoolROIAlign(torch.nn.Module):
    def __init__(self):
        super(ROIPoolROIAlign).__init__()

    def forward(self):
        pass


class HeadClsBox(torch.nn.Module):
    def __init__(self):
        super(HeadClsBox).__init__()

    def forward(self):
        pass


class HeadMask(torch.nn.Module):
    def __init__(self):
        super(HeadMask).__init__()

    def forward(self):
        pass


class MaskRCNN(torch.nn.Module):
    def __init__(self):
        super(MaskRCNN).__init__()

        # 特征提取部分
        self.backbone = torchvision.models.resnet50()
        self.fpn = FPN()

        # RPN部分输出region proposal
        self.rpn = RPN()
        
        # ROIPool+ROIAlign
        self.roi = ROIPoolROIAlign()

        # Mask RCNN的头部网络，做task分支的分化
        self.head_clsbox = HeadClsBox()
        self.head_mask = HeadMask()


    def forward(self):
        pass
