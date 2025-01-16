from __future__ import annotations
import torch.nn as nn
import torch 
from functools import partial

from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrUpBlock
from mamba_ssm import Mamba
import torch.nn.functional as F 

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]

            return x

class MambaLayer(nn.Module):
    def __init__(self, dim, d_state = 16, d_conv = 4, expand = 2, num_slices=None):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
                d_model=dim, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand,    # Block expansion factor
                bimamba_type="v3",
                nslices=num_slices,
        )
    
    def forward(self, x):
        B, C = x.shape[:2]
        assert C == self.dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm)

        out = x_mamba.transpose(-1, -2).reshape(B, C, *img_dims)
        return out
    
class MlpChannel(nn.Module):
    def __init__(self,hidden_size, mlp_dim, ):
        super().__init__()
        self.fc1 = nn.Conv3d(hidden_size, mlp_dim, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv3d(mlp_dim, hidden_size, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class GSC(nn.Module):
    def __init__(self, in_channles) -> None:
        super().__init__()

        self.proj = nn.Conv3d(in_channles, in_channles, 3, 1, 1)
        self.norm = nn.InstanceNorm3d(in_channles)
        self.nonliner = nn.ReLU()

        self.proj2 = nn.Conv3d(in_channles, in_channles, 3, 1, 1)
        self.norm2 = nn.InstanceNorm3d(in_channles)
        self.nonliner2 = nn.ReLU()

        self.proj3 = nn.Conv3d(in_channles, in_channles, 1, 1, 0)
        self.norm3 = nn.InstanceNorm3d(in_channles)
        self.nonliner3 = nn.ReLU()

        self.proj4 = nn.Conv3d(in_channles, in_channles, 1, 1, 0)
        self.norm4 = nn.InstanceNorm3d(in_channles)
        self.nonliner4 = nn.ReLU()

    def forward(self, x):

        x_residual = x 

        x1 = self.proj(x)
        x1 = self.norm(x1)
        x1 = self.nonliner(x1)

        x1 = self.proj2(x1)
        x1 = self.norm2(x1)
        x1 = self.nonliner2(x1)

        x2 = self.proj3(x)
        x2 = self.norm3(x2)
        x2 = self.nonliner3(x2)

        x = x1 + x2
        x = self.proj4(x)
        x = self.norm4(x)
        x = self.nonliner4(x)
        
        return x + x_residual

class MambaEncoder(nn.Module):
    def __init__(self, in_chans=1, depths=[2, 2, 2, 2], dims=[48, 96, 192, 384],
                 drop_path_rate=0., layer_scale_init_value=1e-6, out_indices=[0, 1, 2, 3]):
        super().__init__()

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
              nn.Conv3d(in_chans, dims[0], kernel_size=7, stride=2, padding=3),
              )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                # LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.InstanceNorm3d(dims[i]),
                nn.Conv3d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        self.gscs = nn.ModuleList()
        num_slices_list = [64, 32, 16, 8]
        cur = 0
        for i in range(4):
            gsc = GSC(dims[i])

            stage = nn.Sequential(
                *[MambaLayer(dim=dims[i], num_slices=num_slices_list[i]) for j in range(depths[i])]
            )

            self.stages.append(stage)
            self.gscs.append(gsc)
            cur += depths[i]

        self.out_indices = out_indices

        self.mlps = nn.ModuleList()
        for i_layer in range(4):
            layer = nn.InstanceNorm3d(dims[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)
            self.mlps.append(MlpChannel(dims[i_layer], 2 * dims[i_layer]))

    def forward_features(self, x):
        outs = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.gscs[i](x)
            x = self.stages[i](x)

            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x)
                x_out = self.mlps[i](x_out)
                outs.append(x_out)

        return tuple(outs)

    def forward(self, x):
        x = self.forward_features(x)
        return x

class TSMamba(nn.Module):
    def __init__(
        self,
        in_chans=1,
        out_chans=13,
        depths=[2, 2, 2, 2],
        feat_size=[48, 96, 192, 384],
        drop_path_rate=0,
        layer_scale_init_value=1e-6,
        hidden_size: int = 768,
        norm_name = "instance",
        conv_block: bool = True,
        res_block: bool = True,
        spatial_dims=3,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.depths = depths
        self.drop_path_rate = drop_path_rate
        self.feat_size = feat_size
        self.layer_scale_init_value = layer_scale_init_value

        self.spatial_dims = spatial_dims
        self.vit = MambaEncoder(in_chans, 
                              )
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.in_chans,
            out_channels=self.feat_size[0],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder2 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[0],
            out_channels=self.feat_size[1],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder3 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[1],
            out_channels=self.feat_size[2],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder4 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[2],
            out_channels=self.feat_size[3],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.encoder5 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[3],
            out_channels=self.hidden_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.hidden_size,
            out_channels=self.feat_size[3],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[3],
            out_channels=self.feat_size[2],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[2],
            out_channels=self.feat_size[1],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[1],
            out_channels=self.feat_size[0],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[0],
            out_channels=self.feat_size[0],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=48, out_channels=self.out_chans)

    def proj_feat(self, x):
        new_view = [x.size(0)] + self.proj_view_shape
        x = x.view(new_view)
        x = x.permute(self.proj_axes).contiguous()
        return x

    def forward(self, x_in):
        outs = self.vit(x_in)
        enc1 = self.encoder1(x_in)
        x2 = outs[0]
        enc2 = self.encoder2(x2)
        x3 = outs[1]
        enc3 = self.encoder3(x3)
        x4 = outs[2]
        enc4 = self.encoder4(x4)
        enc_hidden = self.encoder5(outs[3])
        dec3 = self.decoder5(enc_hidden, enc4)
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder2(dec1, enc1)
        out = self.decoder1(dec0) 
        ###0
        # return self.out(out) 
        # ###1 save feats to './feats/pool' 
        # feats = torch.concatenate([F.adaptive_avg_pool3d(enc1,1), F.adaptive_avg_pool3d(enc2,1), F.adaptive_avg_pool3d(enc3,1), \
        #                             F.adaptive_avg_pool3d(enc4,1), F.adaptive_avg_pool3d(enc_hidden,1)], dim=1).squeeze([2,3,4])
        # return self.out(out), feats
        ###2 save feats to './feats/enc_hidden' 
        # feats = enc_hidden.squeeze(0)
        feats = enc_hidden
        return self.out(out), feats
   
class CoAttNet(nn.Module): 
    def __init__( self, in_chans=1, out_chans_class=1, out_chans_surv=10, hidden_size=768) -> None:
        super().__init__()

        self.hidden_size = hidden_size

        self.linear_e = nn.Linear(self.hidden_size, self.hidden_size, bias = False)
        self.gate = nn.Conv3d(self.hidden_size, 1, kernel_size  = 1, bias = False)
        self.gate_s = nn.Sigmoid() 
        self.conv1 = nn.Conv3d(self.hidden_size*2, self.hidden_size, kernel_size=3, padding=1, bias = False, stride=2)
        self.conv2 = nn.Conv3d(self.hidden_size*2, self.hidden_size, kernel_size=3, padding=1, bias = False, stride=2)
        self.bn1 = nn.BatchNorm3d(self.hidden_size)
        self.bn2 = nn.BatchNorm3d(self.hidden_size)
        self.conv3 = nn.Conv3d(self.hidden_size, self.hidden_size//2, kernel_size=3, padding=0, bias = False)
        self.conv4 = nn.Conv3d(self.hidden_size, self.hidden_size//2, kernel_size=3, padding=0, bias = False)
        self.bn3 = nn.BatchNorm3d(self.hidden_size//2)
        self.bn4 = nn.BatchNorm3d(self.hidden_size//2)
        self.prelu = nn.ReLU(inplace=True)

        self.linear_classifier = nn.Linear(self.hidden_size, out_chans_class, bias = False)
        self.linear_survival = nn.Linear(self.hidden_size, out_chans_surv, bias = False)

        # self.main_classifier1 = nn.Conv3d(self.hidden_size, out_chans, kernel_size=1, bias = True)
        # self.main_classifier2 = nn.Conv3d(self.hidden_size, out_chans, kernel_size=1, bias = True)
        self.softmax = nn.Sigmoid()
        
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                #n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
                #init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                #init.xavier_normal(m.weight.data)
                #m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()    

    def forward(self, enc_hidden, enc_hidden2) -> torch.Tensor: 
        ### ref:: https://arxiv.org/pdf/2001.06810.pdf
        N, C, D, H, W = enc_hidden.size()  # Assuming both tensors have the same shape
        all_dim = D * H * W
        fea_size = enc_hidden.size()[2:]    
        enc_hidden2_flat = enc_hidden2.view(-1, enc_hidden.size()[1], all_dim) #N,C,H*W
        # print(enc_hidden2_flat.shape)
        enc_hidden_flat = enc_hidden.view(-1, enc_hidden.size()[1], all_dim)
        # print(enc_hidden_flat.shape)
        enc_hidden2_t = torch.transpose(enc_hidden2_flat,1,2).contiguous()  #batch size x dim x num
        # print(enc_hidden2_t.shape)
        enc_hidden2_corr = self.linear_e(enc_hidden2_t) # 
        A = torch.bmm(enc_hidden2_corr, enc_hidden_flat)
        A1 = F.softmax(A.clone(), dim = 1) #
        B = F.softmax(torch.transpose(A,1,2),dim=1)
        enc_hidden_att = torch.bmm(enc_hidden2_flat, A1).contiguous() #注意我们这个地方要不要用交互以及Residual的结构
        enc_hidden2_att = torch.bmm(enc_hidden_flat, B).contiguous()

        input1_att = enc_hidden_att.view(-1, enc_hidden2.size()[1], fea_size[0], fea_size[1], fea_size[2])  
        input2_att = enc_hidden2_att.view(-1, enc_hidden2.size()[1], fea_size[0], fea_size[1], fea_size[2])
        input1_mask = self.gate(input1_att)
        input2_mask = self.gate(input2_att)
        input1_mask = self.gate_s(input1_mask)
        input2_mask = self.gate_s(input2_mask)
        input1_att = input1_att * input1_mask
        input2_att = input2_att * input2_mask
        input1_att = torch.cat([input1_att, enc_hidden],1) 
        input2_att = torch.cat([input2_att, enc_hidden2],1)

        input1_att  = self.prelu(self.bn1(self.conv1(input1_att ) ) )
        input2_att  = self.prelu(self.bn2(self.conv2(input2_att ) ) ) 
        input1_att  = self.prelu(self.conv3(input1_att ) )
        input2_att  = self.prelu(self.conv4(input2_att ) ) 
        input_att_combine = torch.cat([input1_att, input2_att], 1) 
        input_att_combine = input_att_combine.view(-1, input_att_combine.shape[1]) 
        last = self.linear_classifier(input_att_combine) 
        last = torch.sigmoid(last) 
        last1 = self.linear_survival(input_att_combine) 
        # return last, last1 
        last1 = torch.sigmoid(last1) 
        Regu_weight = [self.linear_survival.weight ]
        return last, [last1, Regu_weight]

class CTSMamba(nn.Module):
    def __init__(self, ) -> None:
        super().__init__()

        self.featsEncoder = TSMamba(in_chans=1,
                                out_chans=1,
                                depths=[2,2,2,2],
                                feat_size=[48, 96, 192, 384]) 
        self.coattNet = CoAttNet(in_chans=1, 
                                out_chans=10,) 
        
        for param in self.featsEncoder.parameters(): 
            param.requires_grad = False 

    def forward(self, x_in1, x_in2): 
        seg1, feats1 = self.featsEncoder(x_in1) 
        seg2, feats2 = self.featsEncoder(x_in2) 
        pred_n, pred_surv = self.coattNet(feats1, feats2) 
        # ###2 save feats to './feats/enc_hidden' 
        # feats = enc_hidden.squeeze(0) 
        # return self.out(out), feats 
        return pred_n, pred_surv 


class CTSMamba_v2(nn.Module):
    def __init__(self, ifloadFeatsEncoder=False, modelFeatsEncoder='./feats/enc_hidden.pth', 
                        ifloadCoAtt=False, iflockweights=True) -> None:
        super().__init__()

        self.featsEncoder = TSMamba(in_chans=1,
                                out_chans=1,
                                depths=[2,2,2,2],
                                feat_size=[48, 96, 192, 384]) 
        self.coattNet = CoAttNet(out_chans_class=1, out_chans_surv=10,) 

        if ifloadFeatsEncoder: self.load_feats_encoder(modelFeatsEncoder)
        if ifloadCoAtt: self.load_co_attention()
        if iflockweights: self.lock_weights()

    def load_feats_encoder(self, modelFeatsEncoder='./feats/enc_hidden.pth'):
        self.featsEncoder.load_state_dict(torch.load(modelFeatsEncoder))
        print("Loaded feature encoder weights.")

    def load_co_attention(self):
        self.coattNet.load_state_dict(torch.load('co_attention_weights.pth'))
        print("Loaded co-attention weights.")

    def lock_weights(self):
        for param in self.featsEncoder.parameters():
            param.requires_grad = False
        print("Feature encoder weights are locked.")

    def forward(self, x_in1, x_in2): 
        seg1, feats1 = self.featsEncoder(x_in1) 
        seg2, feats2 = self.featsEncoder(x_in2) 
        pred_n, pred_surv = self.coattNet(feats1, feats2) 
        # ###2 save feats to './feats/enc_hidden' 
        # feats = enc_hidden.squeeze(0) 
        # return self.out(out), feats 
        return pred_n, pred_surv 



