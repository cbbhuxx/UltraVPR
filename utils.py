import numpy as np
import torch
import random, shutil
import torch.nn as nn
from os.path import join
from models.backbones.e2resnet import E2ResNet
from models.aggregators.se2gem import se2gem
from models.upscale import vlad
import torch.nn as nn


class ultravpr(nn.Module):
    def __init__(self, backbone, aggregator, upscale=None):
        super().__init__()
        self.backbone = backbone
        self.aggregator = aggregator
        self.upscale = upscale
     
    def forward(self, x):
        x = self.backbone(x)
        x = self.aggregator(x)
        if self.upscale is not None:
            x = self.upscale(x)
        return x

def get_aggregator(opt):
    if(opt.pooling.lower() == "se2gem"):
        print('aggregator = se2gem')
        aggregator = se2gem(in_dim=opt.encoder_dim, out_dim=opt.encoder_dim)
    else:
        raise Exception('Unknown aggregators')
    return aggregator

def get_backbone(backbone_model):
    if(backbone_model == "e2resnet50_c8"):
        print('backbone = e2resnet50_c8')
        backbone = E2ResNet(depth=50, out_indices=(3, ), with_geotensor=True, orientation=8, middle_channels=2048) # , orientation=16
        ckpt_path = './pretrain/e2_resnet_50_c8_rot_ours/epoch_100.pth'
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        vpr_model_state_dict = checkpoint['state_dict']
        backbone_state_dict = {k[len('backbone.'):]: v for k, v in vpr_model_state_dict.items() if k.startswith('backbone.')}
        backbone.load_state_dict(backbone_state_dict, strict=False)
    
    elif(backbone_model == "e2resnet50_c16"):
        print('backbone = e2resnet50_c16')
        backbone = E2ResNet(depth=50, out_indices=(3, ), with_geotensor=True,orientation=16,  middle_channels=1024)
        ckpt_path = './pretrain/e2_resnet_50_c16_rot_ours/epoch_100.pth'
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        vpr_model_state_dict = checkpoint['state_dict']
        backbone_state_dict = {k[len('backbone.'):]: v for k, v in vpr_model_state_dict.items() if k.startswith('backbone.')}
        backbone.load_state_dict(backbone_state_dict, strict=False)

    else:
        raise Exception('Unknown backbone')

    return backbone
def set_model(opt):

    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    torch.cuda.manual_seed(123)

    backbone = get_backbone(opt.backbone)
    aggregator = get_aggregator(opt)
    
    upscale = None
    if opt.upscaling:
        num_all = sum(opt.num_classes)
        upscale = vlad(num_all, opt.encoder_dim)
    if upscale is not None:
        model = ultravpr(backbone, aggregator, upscale)
    else:
        model = ultravpr(backbone, aggregator)
        
    # 加载已训练训练参数
    if opt.resume:
        resume_ckpt = join(opt.resume, 'checkpoints', 'model_best.pth.tar')
        checkpoint = torch.load(resume_ckpt, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        print("=> loaded checkpoint '{}'".format(resume_ckpt))

    return model

def save_checkpoint(opt,state, is_best, filename='checkpoint.pth.tar'):
    model_out_path = join(opt.savePath, filename)
    torch.save(state, model_out_path)
    if is_best:
        shutil.copyfile(model_out_path, join(opt.savePath, 'model_best.pth.tar'))