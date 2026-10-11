import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose

from .dinov2 import DINOv2
from .util.blocks import FeatureFusionBlock, _make_scratch
from .util.transform import Resize, NormalizeImage, PrepareForNet


def _make_fusion_block(features, use_bn, size=None):
    return FeatureFusionBlock(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
    )


class ConvBlock(nn.Module):
    def __init__(self, in_feature, out_feature):
        super().__init__()
        
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_feature, out_feature, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_feature),
            nn.ReLU(True)
        )
    
    def forward(self, x):
        return self.conv_block(x)


class DPTHead(nn.Module):
    def __init__(
        self, 
        in_channels, 
        features=256, 
        use_bn=False, 
        out_channels=[256, 512, 1024, 1024], 
        use_clstoken=False
    ):
        super(DPTHead, self).__init__()
        
        self.use_clstoken = use_clstoken
        
        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for out_channel in out_channels
        ])
        
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(
                in_channels=out_channels[0],
                out_channels=out_channels[0],
                kernel_size=4,
                stride=4,
                padding=0),
            nn.ConvTranspose2d(
                in_channels=out_channels[1],
                out_channels=out_channels[1],
                kernel_size=2,
                stride=2,
                padding=0),
            nn.Identity(),
            nn.Conv2d(
                in_channels=out_channels[3],
                out_channels=out_channels[3],
                kernel_size=3,
                stride=2,
                padding=1)
        ])
        
        if use_clstoken:
            self.readout_projects = nn.ModuleList()
            for _ in range(len(self.projects)):
                self.readout_projects.append(
                    nn.Sequential(
                        nn.Linear(2 * in_channels, in_channels),
                        nn.GELU()))
        
        self.scratch = _make_scratch(
            out_channels,
            features,
            groups=1,
            expand=False,
        )
        
        self.scratch.stem_transpose = None
        
        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)
        
        head_features_1 = features
        head_features_2 = 32
        
        self.scratch.output_conv1 = nn.Conv2d(head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1)
        self.scratch.output_conv2 = nn.Sequential(
            nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(head_features_2, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True),
            nn.Identity(),
        )
    
    def forward(self, out_features, patch_h, patch_w):
        out = []
        for i, x in enumerate(out_features):
            if self.use_clstoken:
                x, cls_token = x[0], x[1]
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))
            else:
                x = x[0]
            
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
            
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            
            out.append(x)
        
        layer_1, layer_2, layer_3, layer_4 = out
        
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)
        
        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])        
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
        
        out = self.scratch.output_conv1(path_1)
        out = F.interpolate(out, (int(patch_h * 14), int(patch_w * 14)), mode="bilinear", align_corners=True)
        out = self.scratch.output_conv2(out)
        
        return out


class DepthAnythingV2(nn.Module):
    def __init__(
        self, 
        encoder='vitl', 
        features=256, 
        out_channels=[256, 512, 1024, 1024], 
        use_bn=False, 
        use_clstoken=False
    ):
        super(DepthAnythingV2, self).__init__()
        
        self.intermediate_layer_idx = {
            'vits': [2, 5, 8, 11],
            'vitb': [2, 5, 8, 11], 
            'vitl': [4, 11, 17, 23], 
            'vitg': [9, 19, 29, 39]
        }
        
        self.encoder = encoder
        self.pretrained = DINOv2(model_name=encoder)
        
        self.depth_head = DPTHead(self.pretrained.embed_dim, features, use_bn, out_channels=out_channels, use_clstoken=use_clstoken)
    
    def forward(self, x):
        patch_h, patch_w = x.shape[-2] // 14, x.shape[-1] // 14
        
        features = self.pretrained.get_intermediate_layers(x, self.intermediate_layer_idx[self.encoder], return_class_token=True)
        
        depth = self.depth_head(features, patch_h, patch_w)
        depth = F.relu(depth)
        
        return depth.squeeze(1)
    
    @torch.no_grad()
    def infer_image(self, raw_image, input_size=518):
        image, (h, w) = self.image2tensor(raw_image, input_size)
        
        depth = self.forward(image)
        
        depth = F.interpolate(depth[:, None], (h, w), mode="bilinear", align_corners=True)[0, 0]
        
        return depth.cpu().numpy()

    @staticmethod
    def _prepare_project_rgb_for_dinov2(
        images: torch.Tensor, model_dtype: torch.dtype
    ) -> torch.Tensor:
        """Convert ApDepth RGB tensors to the input domain expected by DINOv2.

        ApDepth creates ``rgb_norm`` in ``src/dataset/base_depth_dataset.py`` as
        ``rgb / 255 * 2 - 1``, so this method receives RGB in [-1, 1].  The
        official DA2 ``image2tensor`` path below instead starts from RGB in
        [0, 1] and applies ImageNet mean/std normalization.  Passing ApDepth's
        [-1, 1] tensor directly to DINOv2 skips both conversions and is the
        input-range mismatch fixed here.
        """
        if not images.is_floating_point():
            raise TypeError(f"Expected floating-point RGB, got {images.dtype}")

        images_01 = ((images.to(dtype=model_dtype) + 1.0) * 0.5).clamp_(0.0, 1.0)
        mean = images_01.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = images_01.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        return (images_01 - mean) / std

    @staticmethod
    def _relative_disparity_to_apdepth_range(
        disparity: torch.Tensor, output_dtype: torch.dtype
    ) -> torch.Tensor:
        """Map DA2 relative disparity to ApDepth's depth/VAE convention.

        DA2's relative model emits an unbounded disparity-like value: larger is
        nearer.  ApDepth's ``ScaleShiftDepthNormalizer`` uses [-1, 1] with
        ``far_plane_at_max=True``: -1 is near and +1 is far.  SD2's VAE also
        expects an image-like tensor in [-1, 1].  Returning raw DA2 output from
        ``infer_batch`` therefore mismatched both its direction and range.

        The same 2%/98% robust per-image normalization used for the training GT
        removes DA2's arbitrary relative scale.  The subtraction from one then
        changes disparity (near-is-large) into depth (far-is-large).
        """
        disparity_fp32 = disparity.float()
        flat = disparity_fp32.flatten(1)
        low_q = torch.quantile(flat, 0.02, dim=1).view(-1, 1, 1, 1)
        high_q = torch.quantile(flat, 0.98, dim=1).view(-1, 1, 1, 1)
        disparity_01 = (disparity_fp32 - low_q) / (
            high_q - low_q
        ).clamp_min(1e-6)
        disparity_01 = disparity_01.clamp_(0.0, 1.0)

        depth_01 = 1.0 - disparity_01
        depth_m11 = depth_01.mul(2.0).sub(1.0)
        return depth_m11.to(dtype=output_dtype)
    
    @torch.no_grad()
    def infer_batch(self, images: torch.Tensor):
        """Build a three-channel DA2 prior compatible with ApDepth's SD2 VAE.

        Args:
            images: ApDepth RGB tensor [B, 3, H, W] in [-1, 1].

        Returns:
            Relative depth [B, 3, H, W] in [-1, 1], where -1 is near and +1
            is far.  This is a relative prior; it is not metric depth in metres.
        """
        assert images.ndim == 4 and images.size(1) == 3, \
            f"Expect input shape [B,3,H,W], but got {images.shape}"

        model_parameter = next(self.parameters())
        device = model_parameter.device
        output_dtype = images.dtype
        # Resize in fp32 because CUDA antialiased bicubic kernels are not
        # available for every fp16/bf16 PyTorch build. DINO input is cast to
        # the actual DA2 weight dtype after resizing.
        images = images.to(device=device, dtype=torch.float32)

        _, _, H, W = images.shape

        new_H = int((H + 13) // 14 * 14)
        new_W = int((W + 13) // 14 * 14)

        if new_H != H or new_W != W:
            images = F.interpolate(
                images,
                size=(new_H, new_W),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )

        # RANGE CONTRACT 1: ApDepth [-1, 1] RGB -> DA2/DINOv2 ImageNet input.
        images_dinov2 = self._prepare_project_rgb_for_dinov2(
            images, model_dtype=model_parameter.dtype
        )

        patch_h, patch_w = new_H // 14, new_W // 14

        features = self.pretrained.get_intermediate_layers(
            images_dinov2,
            self.intermediate_layer_idx[self.encoder],
            return_class_token=True,
        )

        # DA2 relative output is disparity-like (near is large), not metric depth.
        disparity = F.relu(self.depth_head(features, patch_h, patch_w))
        disparity = F.interpolate(
            disparity, size=(H, W), mode="bilinear", align_corners=False
        )

        # RANGE CONTRACT 2: raw DA2 disparity -> ApDepth/VAE depth in [-1, 1].
        depth_m11 = self._relative_disparity_to_apdepth_range(
            disparity, output_dtype=output_dtype
        )

        return depth_m11.repeat(1, 3, 1, 1)

    
    
    def image2tensor(self, raw_image, input_size=518):        
        transform = Compose([
            Resize(
                width=input_size,
                height=input_size,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])
        
        h, w = raw_image.shape[:2]
        
        image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0
        
        image = transform({'image': image})['image']
        image = torch.from_numpy(image).unsqueeze(0)
        
        DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        image = image.to(DEVICE)
        
        return image, (h, w)
