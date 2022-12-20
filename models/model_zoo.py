import torch
import torch.nn as nn
import pdb

# ====================================
# Normalizer class
# ====================================
class Normalizer(nn.Module):
    def __init__(self, exp_config):
        super(Normalizer, self).__init__()
        num_layers = exp_config.norm_num_hidden_layers
        n1 = exp_config.norm_num_filters_per_layer
        k = exp_config.norm_kernel_size
        
        self.exp_config = exp_config
        self.conv = nn.ModuleList()
        self.bn = nn.ModuleList()
        n0 = 1
        for l in range(num_layers):
            self.conv.append(nn.Conv2d(n0, n1, kernel_size = k, padding = 1))
            n0 = n1
            
            if exp_config.norm_batch_norm is True:
                self.bn.append(nn.BatchNorm2d(n1))
        
        self.delta = nn.Conv2d(n1, 1, kernel_size = k, padding = 1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        num_layers = self.exp_config.norm_num_hidden_layers
        n1 = self.exp_config.norm_num_filters_per_layer
        
        out = x
        for l in range(num_layers):
            out = self.conv[l](out)
            
            if self.exp_config.norm_batch_norm is True:
                out = self.bn[l](out)
            
            if self.exp_config.norm_activation == 'elu':
                out = nn.ELU()(out)
            elif self.exp_config.norm_activation == 'relu':
                out = nn.ReLU()(out)
            elif self.exp_config.norm_activation == 'rbf':
                scale = torch.randn(size = [n1, 1, 1, 1], requires_grad = True).cuda() * 0.05 + 0.2
                out = torch.exp(-(out**2) / (scale**2))
        
        out = self.delta(out)
        out = x + out
        
        return out

# ====================================
# Encoder class - no skip connection
# ====================================
class Encoder(nn.Module):
    def __init__(self, n0):
        super(Encoder, self).__init__()
        n1, n2, n3, n4 = 1*n0, 2*n0, 4*n0, 8*n0 # number of channels
        
        # ====================================
        # 1st Conv block - two conv layers, followed by batch normalization and max-pooling
        # ====================================
        self.conv1_1 = nn.Conv2d(1, n1, kernel_size = 3, padding = 1)
        self.conv1_2 = nn.Conv2d(n1, n1, kernel_size = 3, padding = 1)
        self.bn1_1 = nn.BatchNorm2d(n1)
        self.bn1_2 = nn.BatchNorm2d(n1)
        
        # ====================================
        # 2nd Conv block
        # ====================================
        self.conv2_1 = nn.Conv2d(n1, n2, kernel_size = 3, padding = 1)
        self.conv2_2 = nn.Conv2d(n2, n2, kernel_size = 3, padding = 1)
        self.bn2_1 = nn.BatchNorm2d(n2)
        self.bn2_2 = nn.BatchNorm2d(n2)
        
        # ====================================
        # 3rd Conv block
        # ====================================
        self.conv3_1 = nn.Conv2d(n2, n3, kernel_size = 3, padding = 1)
        self.conv3_2 = nn.Conv2d(n3, n3, kernel_size = 3, padding = 1)
        self.bn3_1 = nn.BatchNorm2d(n3)
        self.bn3_2 = nn.BatchNorm2d(n3)
        
        # ====================================
        # 4th Conv block
        # ====================================
        self.conv4_1 = nn.Conv2d(n3, n4, kernel_size = 3, padding = 1)
        self.conv4_2 = nn.Conv2d(n4, n4, kernel_size = 3, padding = 1)
        self.bn4_1 = nn.BatchNorm2d(n4)
        self.bn4_2 = nn.BatchNorm2d(n4)
        
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.max_pool = nn.MaxPool2d(2)
    
    def forward(self, x):
        # ====================================
        # 1st Conv block - two conv layers, followed by batch normalization and max-pooling
        # ====================================
        conv1_1 = self.relu(self.bn1_1(self.conv1_1(x)))
        conv1_2 = self.relu(self.bn1_2(self.conv1_2(conv1_1)))
        pool1 = self.max_pool(conv1_2)
        
        # ====================================
        # 2nd Conv block
        # ====================================
        conv2_1 = self.relu(self.bn2_1(self.conv2_1(pool1)))
        conv2_2 = self.relu(self.bn2_2(self.conv2_2(conv2_1)))
        pool2 = self.max_pool(conv2_2)
        
        # ====================================
        # 3rd Conv block
        # ====================================
        conv3_1 = self.relu(self.bn3_1(self.conv3_1(pool2)))
        conv3_2 = self.relu(self.bn3_2(self.conv3_2(conv3_1)))
        pool3 = self.max_pool(conv3_2)
        
        # ====================================
        # 4th Conv block
        # ====================================
        conv4_1 = self.relu(self.bn4_1(self.conv4_1(pool3)))
        conv4_2 = self.relu(self.bn4_2(self.conv4_2(conv4_1)))
        
        return conv4_2

# ====================================
# Decoder class - no skip connection
# ====================================    
class Decoder(nn.Module):
    def __init__(self, n0, num_classes):
        super(Decoder, self).__init__()
        n1, n2, n3, n4 = 1*n0, 2*n0, 4*n0, 8*n0 # number of channels
        
        # ====================================
        # Upsampling via bilinear upsampling, concatenation (skip connection), followed by 2 conv layers
        # ====================================
        self.deconv3 = nn.Upsample(scale_factor = 2, mode = 'bilinear', align_corners = True)
        self.conv5_1 = nn.Conv2d(n4, n3, kernel_size = 3, padding = 1)
        self.conv5_2 = nn.Conv2d(n3, n3, kernel_size = 3, padding = 1)
        self.bn5_1 = nn.BatchNorm2d(n3)
        self.bn5_2 = nn.BatchNorm2d(n3)
        
        # ====================================
        # Upsampling via bilinear upsampling, concatenation (skip connection), followed by 2 conv layers
        # ====================================
        self.deconv2 = nn.Upsample(scale_factor = 2, mode = 'bilinear', align_corners = True)
        self.conv6_1 = nn.Conv2d(n3, n2, kernel_size = 3, padding = 1)
        self.conv6_2 = nn.Conv2d(n2, n2, kernel_size = 3, padding = 1)
        self.bn6_1 = nn.BatchNorm2d(n2)
        self.bn6_2 = nn.BatchNorm2d(n2)
        
        # ====================================
        # Upsampling via bilinear upsampling, concatenation (skip connection), followed by 2 conv layers
        # ====================================
        self.deconv1 = nn.Upsample(scale_factor = 2, mode = 'bilinear', align_corners = True)
        self.conv7_1 = nn.Conv2d(n2, n1, kernel_size = 3, padding = 1)
        self.conv7_2 = nn.Conv2d(n1, n1, kernel_size = 3, padding = 1)
        self.bn7_1 = nn.BatchNorm2d(n1)
        self.bn7_2 = nn.BatchNorm2d(n1)
        
        # ====================================
        # Final conv layer - without batch normalization or activation
        # ====================================
        self.conv_pred = nn.Conv2d(n1, num_classes, kernel_size = 3, padding = 1, bias = False)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.max_pool = nn.MaxPool2d(2)
        
    def forward(self, zs):
        conv4_2 = zs
        
        # ====================================
        # Upsampling via bilinear upsampling, followed by 2 conv layers
        # ====================================
        deconv3 = self.deconv3(conv4_2)
        conv5_1 = self.relu(self.bn5_1(self.conv5_1(deconv3)))
        conv5_2 = self.relu(self.bn5_2(self.conv5_2(conv5_1)))
        
        # ====================================
        # Upsampling via bilinear upsampling, followed by 2 conv layers
        # ====================================
        deconv2 = self.deconv2(conv5_2)
        conv6_1 = self.relu(self.bn6_1(self.conv6_1(deconv2)))
        conv6_2 = self.relu(self.bn6_2(self.conv6_2(conv6_1)))
        
        # ====================================
        # Upsampling via bilinear upsampling, followed by 2 conv layers
        # ====================================
        deconv1 = self.deconv1(conv6_2)
        conv7_1 = self.relu(self.bn7_1(self.conv7_1(deconv1)))
        conv7_2 = self.relu(self.bn7_2(self.conv7_2(conv7_1)))
        
        # ====================================
        # Final conv layer - without batch normalization or activation
        # ====================================
        pred = self.conv_pred(conv7_2)
        
        return pred
    
# ====================================
# Encoder class for UNet
# ====================================
class EncoderUNet(nn.Module):
    def __init__(self, n0):
        super(EncoderUNet, self).__init__()
        n1, n2, n3, n4 = 1*n0, 2*n0, 4*n0, 8*n0 # number of channels
        
        # ====================================
        # 1st Conv block - two conv layers, followed by batch normalization and max-pooling
        # ====================================
        self.conv1_1 = nn.Conv2d(1, n1, kernel_size = 3, padding = 1)
        self.conv1_2 = nn.Conv2d(n1, n1, kernel_size = 3, padding = 1)
        self.bn1_1 = nn.BatchNorm2d(n1)
        self.bn1_2 = nn.BatchNorm2d(n1)
        
        # ====================================
        # 2nd Conv block
        # ====================================
        self.conv2_1 = nn.Conv2d(n1, n2, kernel_size = 3, padding = 1)
        self.conv2_2 = nn.Conv2d(n2, n2, kernel_size = 3, padding = 1)
        self.bn2_1 = nn.BatchNorm2d(n2)
        self.bn2_2 = nn.BatchNorm2d(n2)
        
        # ====================================
        # 3rd Conv block
        # ====================================
        self.conv3_1 = nn.Conv2d(n2, n3, kernel_size = 3, padding = 1)
        self.conv3_2 = nn.Conv2d(n3, n3, kernel_size = 3, padding = 1)
        self.bn3_1 = nn.BatchNorm2d(n3)
        self.bn3_2 = nn.BatchNorm2d(n3)
        
        # ====================================
        # 4th Conv block
        # ====================================
        self.conv4_1 = nn.Conv2d(n3, n4, kernel_size = 3, padding = 1)
        self.conv4_2 = nn.Conv2d(n4, n4, kernel_size = 3, padding = 1)
        self.bn4_1 = nn.BatchNorm2d(n4)
        self.bn4_2 = nn.BatchNorm2d(n4)
        
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.max_pool = nn.MaxPool2d(2)
        
    def forward(self, x):
        # ====================================
        # 1st Conv block - two conv layers, followed by batch normalization and max-pooling
        # ====================================
        conv1_1 = self.relu(self.bn1_1(self.conv1_1(x)))
        conv1_2 = self.relu(self.bn1_2(self.conv1_2(conv1_1)))
        pool1 = self.max_pool(conv1_2)
        
        # ====================================
        # 2nd Conv block
        # ====================================
        conv2_1 = self.relu(self.bn2_1(self.conv2_1(pool1)))
        conv2_2 = self.relu(self.bn2_2(self.conv2_2(conv2_1)))
        pool2 = self.max_pool(conv2_2)
        
        # ====================================
        # 3rd Conv block
        # ====================================
        conv3_1 = self.relu(self.bn3_1(self.conv3_1(pool2)))
        conv3_2 = self.relu(self.bn3_2(self.conv3_2(conv3_1)))
        pool3 = self.max_pool(conv3_2)
        
        # ====================================
        # 4th Conv block
        # ====================================
        conv4_1 = self.relu(self.bn4_1(self.conv4_1(pool3)))
        conv4_2 = self.relu(self.bn4_2(self.conv4_2(conv4_1)))
        
        return conv1_2, conv2_2, conv3_2, conv4_2
    
# ====================================
# Decoder class for UNet
# ====================================    
class DecoderUNet(nn.Module):
    def __init__(self, n0, num_classes):
        super(DecoderUNet, self).__init__()
        n1, n2, n3, n4 = 1*n0, 2*n0, 4*n0, 8*n0 # number of channels
        
        # ====================================
        # Upsampling via bilinear upsampling, concatenation (skip connection), followed by 2 conv layers
        # ====================================
        self.deconv3 = nn.Upsample(scale_factor = 2, mode = 'bilinear', align_corners = True)
        self.conv5_1 = nn.Conv2d(n4 + n3, n3, kernel_size = 3, padding = 1)
        self.conv5_2 = nn.Conv2d(n3, n3, kernel_size = 3, padding = 1)
        self.bn5_1 = nn.BatchNorm2d(n3)
        self.bn5_2 = nn.BatchNorm2d(n3)
        
        # ====================================
        # Upsampling via bilinear upsampling, concatenation (skip connection), followed by 2 conv layers
        # ====================================
        self.deconv2 = nn.Upsample(scale_factor = 2, mode = 'bilinear', align_corners = True)
        self.conv6_1 = nn.Conv2d(n3 + n2, n2, kernel_size = 3, padding = 1)
        self.conv6_2 = nn.Conv2d(n2, n2, kernel_size = 3, padding = 1)
        self.bn6_1 = nn.BatchNorm2d(n2)
        self.bn6_2 = nn.BatchNorm2d(n2)
        
        # ====================================
        # Upsampling via bilinear upsampling, concatenation (skip connection), followed by 2 conv layers
        # ====================================
        self.deconv1 = nn.Upsample(scale_factor = 2, mode = 'bilinear', align_corners = True)
        self.conv7_1 = nn.Conv2d(n2 + n1, n1, kernel_size = 3, padding = 1)
        self.conv7_2 = nn.Conv2d(n1, n1, kernel_size = 3, padding = 1)
        self.bn7_1 = nn.BatchNorm2d(n1)
        self.bn7_2 = nn.BatchNorm2d(n1)
        
        # ====================================
        # Final conv layer - without batch normalization or activation
        # ====================================
        self.conv_pred = nn.Conv2d(n1, num_classes, kernel_size = 3, padding = 1, bias = False)
        
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.max_pool = nn.MaxPool2d(2)
        
    def forward(self, zs):
        conv1_2, conv2_2, conv3_2, conv4_2 = zs
        
        # ====================================
        # Upsampling via bilinear upsampling, concatenation (skip connection), followed by 2 conv layers
        # ====================================
        deconv3 = self.deconv3(conv4_2)
        skip3 = torch.cat([deconv3, conv3_2], axis = 1)
        conv5_1 = self.relu(self.bn5_1(self.conv5_1(skip3)))
        conv5_2 = self.relu(self.bn5_2(self.conv5_2(conv5_1)))
        
        # ====================================
        # Upsampling via bilinear upsampling, concatenation (skip connection), followed by 2 conv layers
        # ====================================
        deconv2 = self.deconv2(conv5_2)
        skip2 = torch.cat([deconv2, conv2_2], axis = 1)
        conv6_1 = self.relu(self.bn6_1(self.conv6_1(skip2)))
        conv6_2 = self.relu(self.bn6_2(self.conv6_2(conv6_1)))
        
        # ====================================
        # Upsampling via bilinear upsampling, concatenation (skip connection), followed by 2 conv layers
        # ====================================
        deconv1 = self.deconv1(conv6_2)
        skip1 = torch.cat([deconv1, conv1_2], axis = 1)
        conv7_1 = self.relu(self.bn7_1(self.conv7_1(skip1)))
        conv7_2 = self.relu(self.bn7_2(self.conv7_2(conv7_1)))
        
        # ====================================
        # Final conv layer - without batch normalization or activation
        # ====================================
        pred = self.conv_pred(conv7_2)
        
        return pred
    

# ====================================
# Encoder class for 3D-UNet 
# ====================================
class EncoderUNet3D(nn.Module):
    def __init__(self, n0):
        super(EncoderUNet3D, self).__init__()
        n1, n2, n3, n4 = 1*n0, 2*n0, 4*n0, 8*n0 # number of channels
        
        
        # ====================================
        # 1st Conv block - two conv layers, followed by batch normalization and max-pooling
        # ====================================
        self.conv1_1 = nn.Conv3d(1, n1, kernel_size = 3, padding = 1)
        self.conv1_2 = nn.Conv3d(n1, n1, kernel_size = 3, padding = 1)
        self.conv1_3 = nn.Conv3d(n1, n1, kernel_size = 3, padding = 1)
        self.bn1_1 = nn.BatchNorm3d(n1)
        self.bn1_2 = nn.BatchNorm3d(n1)
        
        # ====================================
        # 2nd Conv block
        # ====================================
        self.conv2_1 = nn.Conv3d(n1, n2, kernel_size = 3, padding = 1)
        self.conv2_2 = nn.Conv3d(n2, n2, kernel_size = 3, padding = 1)
        self.bn2_1 = nn.BatchNorm3d(n2)
        self.bn2_2 = nn.BatchNorm3d(n2)
        
        # ====================================
        # 3rd Conv block
        # ====================================
        self.conv3_1 = nn.Conv3d(n2, n3, kernel_size = 3, padding = 1)
        self.conv3_2 = nn.Conv3d(n3, n3, kernel_size = 3, padding = 1)
        self.bn3_1 = nn.BatchNorm3d(n3)
        self.bn3_2 = nn.BatchNorm3d(n3)
        """
        # ====================================
        # 4th Conv block
        # ====================================
        self.conv4_1 = nn.Conv3d(n3, n4, kernel_size = 3, padding = 1)
        self.conv4_2 = nn.Conv3d(n4, n4, kernel_size = 3, padding = 1)
        self.bn4_1 = nn.BatchNorm3d(n4)
        self.bn4_2 = nn.BatchNorm3d(n4)
        """
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.max_pool = nn.MaxPool3d(2)
        
    def forward(self, x):
        # ====================================
        # 1st Conv block - two conv layers, followed by batch normalization and max-pooling
        # ====================================
        conv1_1 = self.relu(self.bn1_1(self.conv1_1(x)))
        conv1_2 = self.relu(self.bn1_2(self.conv1_2(conv1_1)))
        pool1 = self.max_pool(conv1_2)
        
        # ====================================
        # 2nd Conv block
        # ====================================
        conv2_1 = self.relu(self.bn2_1(self.conv2_1(pool1)))
        conv2_2 = self.relu(self.bn2_2(self.conv2_2(conv2_1)))
        pool2 = self.max_pool(conv2_2)
        
        # ====================================
        # 3rd Conv block
        # ====================================
        conv3_1 = self.relu(self.bn3_1(self.conv3_1(pool2)))
        conv3_2 = self.relu(self.bn3_2(self.conv3_2(conv3_1)))
        #pool3 = self.max_pool(conv3_2)
        """
        # ====================================
        # 4th Conv block
        # ====================================
        conv4_1 = self.relu(self.bn4_1(self.conv4_1(pool3)))
        conv4_2 = self.relu(self.bn4_2(self.conv4_2(conv4_1)))
        """
        return conv1_2, conv2_2, conv3_2#, conv4_2
    
# ====================================
# Decoder class for 3D-UNet
# ====================================    
class DecoderUNet3D(nn.Module):
    def __init__(self, n0, num_classes):
        super(DecoderUNet3D, self).__init__()
        n1, n2, n3, n4 = 1*n0, 2*n0, 4*n0, 8*n0 # number of channels
        """
        # ====================================
        # Upsampling via bilinear upsampling, concatenation (skip connection), followed by 2 conv layers
        # ====================================
        self.deconv3 = nn.Upsample(scale_factor = 2, mode = 'trilinear', align_corners = True)
        self.conv5_1 = nn.Conv3d(n4 + n3, n3, kernel_size = 3, padding = 1)
        self.conv5_2 = nn.Conv3d(n3, n3, kernel_size = 3, padding = 1)
        self.bn5_1 = nn.BatchNorm3d(n3)
        self.bn5_2 = nn.BatchNorm3d(n3)
        """
        # ====================================
        # Upsampling via bilinear upsampling, concatenation (skip connection), followed by 2 conv layers
        # ====================================
        self.deconv2 = nn.Upsample(scale_factor = 2, mode = 'trilinear', align_corners = True)
        self.conv6_1 = nn.Conv3d(n3 + n2, n2, kernel_size = 3, padding = 1)
        self.conv6_2 = nn.Conv3d(n2, n2, kernel_size = 3, padding = 1)
        self.bn6_1 = nn.BatchNorm3d(n2)
        self.bn6_2 = nn.BatchNorm3d(n2)
        
        # ====================================
        # Upsampling via bilinear upsampling, concatenation (skip connection), followed by 2 conv layers
        # ====================================
        self.deconv1 = nn.Upsample(scale_factor = 2, mode = 'trilinear', align_corners = True)
        self.conv7_1 = nn.Conv3d(n2 + n1, n1, kernel_size = 3, padding = 1)
        self.conv7_2 = nn.Conv3d(n1, n1, kernel_size = 3, padding = 1)
        self.bn7_1 = nn.BatchNorm3d(n1)
        self.bn7_2 = nn.BatchNorm3d(n1)
        
        # ====================================
        # Final conv layer - without batch normalization or activation
        # ====================================
        self.conv_pred = nn.Conv3d(n1, num_classes, kernel_size = 3, padding = 1, bias = False)
        
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.max_pool = nn.MaxPool3d(2)
        
    def forward(self, zs):
        #conv1_2, conv2_2, conv3_2, conv4_2 = zs
        conv1_2, conv2_2, conv3_2 = zs
        """
        # ====================================
        # Upsampling via bilinear upsampling, concatenation (skip connection), followed by 2 conv layers
        # ====================================
        deconv3 = self.deconv3(conv4_2)
        skip3 = torch.cat([deconv3, conv3_2], axis = 1)
        conv5_1 = self.relu(self.bn5_1(self.conv5_1(skip3)))
        conv5_2 = self.relu(self.bn5_2(self.conv5_2(conv5_1)))
        """
        # ====================================
        # Upsampling via bilinear upsampling, concatenation (skip connection), followed by 2 conv layers
        # ====================================
        deconv2 = self.deconv2(conv3_2)
        skip2 = torch.cat([deconv2, conv2_2], axis = 1)
        conv6_1 = self.relu(self.bn6_1(self.conv6_1(skip2)))
        conv6_2 = self.relu(self.bn6_2(self.conv6_2(conv6_1)))
        
        # ====================================
        # Upsampling via bilinear upsampling, concatenation (skip connection), followed by 2 conv layers
        # ====================================
        deconv1 = self.deconv1(conv6_2)
        skip1 = torch.cat([deconv1, conv1_2], axis = 1)
        conv7_1 = self.relu(self.bn7_1(self.conv7_1(skip1)))
        conv7_2 = self.relu(self.bn7_2(self.conv7_2(conv7_1)))
        
        # ====================================
        # Final conv layer - without batch normalization or activation
        # ====================================
        pred = self.conv_pred(conv7_2)
        
        return pred



# ====================================
# Encoder class for 3D-UNet 
# ====================================
class EncoderUNet3D(nn.Module):
    def __init__(self, n0):
        super(EncoderUNet3D, self).__init__()
        n1, n2, n3, n4 = 1*n0, 2*n0, 4*n0, 8*n0 # number of channels
        
        
        # ====================================
        # 1st Conv block - two conv layers, followed by batch normalization and max-pooling
        # ====================================
        self.conv1_1 = nn.Conv3d(1, n1, kernel_size = 3, padding = 1)
        self.conv1_2 = nn.Conv3d(n1, n1, kernel_size = 3, padding = 1)
        self.conv1_3 = nn.Conv3d(n1, n1, kernel_size = 3, padding = 1)
        self.bn1_1 = nn.BatchNorm3d(n1)
        self.bn1_2 = nn.BatchNorm3d(n1)
        
        # ====================================
        # 2nd Conv block
        # ====================================
        self.conv2_1 = nn.Conv3d(n1, n2, kernel_size = 3, padding = 1)
        self.conv2_2 = nn.Conv3d(n2, n2, kernel_size = 3, padding = 1)
        self.bn2_1 = nn.BatchNorm3d(n2)
        self.bn2_2 = nn.BatchNorm3d(n2)
        
        # ====================================
        # 3rd Conv block
        # ====================================
        self.conv3_1 = nn.Conv3d(n2, n3, kernel_size = 3, padding = 1)
        self.conv3_2 = nn.Conv3d(n3, n3, kernel_size = 3, padding = 1)
        self.bn3_1 = nn.BatchNorm3d(n3)
        self.bn3_2 = nn.BatchNorm3d(n3)
        """
        # ====================================
        # 4th Conv block
        # ====================================
        self.conv4_1 = nn.Conv3d(n3, n4, kernel_size = 3, padding = 1)
        self.conv4_2 = nn.Conv3d(n4, n4, kernel_size = 3, padding = 1)
        self.bn4_1 = nn.BatchNorm3d(n4)
        self.bn4_2 = nn.BatchNorm3d(n4)
        """
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.max_pool = nn.MaxPool3d(2)
        
    def forward(self, x):
        # ====================================
        # 1st Conv block - two conv layers, followed by batch normalization and max-pooling
        # ====================================
        conv1_1 = self.relu(self.bn1_1(self.conv1_1(x)))
        conv1_2 = self.relu(self.bn1_2(self.conv1_2(conv1_1)))
        pool1 = self.max_pool(conv1_2)
        
        # ====================================
        # 2nd Conv block
        # ====================================
        conv2_1 = self.relu(self.bn2_1(self.conv2_1(pool1)))
        conv2_2 = self.relu(self.bn2_2(self.conv2_2(conv2_1)))
        pool2 = self.max_pool(conv2_2)
        
        # ====================================
        # 3rd Conv block
        # ====================================
        conv3_1 = self.relu(self.bn3_1(self.conv3_1(pool2)))
        conv3_2 = self.relu(self.bn3_2(self.conv3_2(conv3_1)))
        #pool3 = self.max_pool(conv3_2)
        """
        # ====================================
        # 4th Conv block
        # ====================================
        conv4_1 = self.relu(self.bn4_1(self.conv4_1(pool3)))
        conv4_2 = self.relu(self.bn4_2(self.conv4_2(conv4_1)))
        """
        return conv1_2, conv2_2, conv3_2#, conv4_2
    
# ====================================
# Decoder class for 3D-UNet
# ====================================    
class DecoderUNet3D(nn.Module):
    def __init__(self, n0, num_classes):
        super(DecoderUNet3D, self).__init__()
        n1, n2, n3, n4 = 1*n0, 2*n0, 4*n0, 8*n0 # number of channels
        """
        # ====================================
        # Upsampling via bilinear upsampling, concatenation (skip connection), followed by 2 conv layers
        # ====================================
        self.deconv3 = nn.Upsample(scale_factor = 2, mode = 'trilinear', align_corners = True)
        self.conv5_1 = nn.Conv3d(n4 + n3, n3, kernel_size = 3, padding = 1)
        self.conv5_2 = nn.Conv3d(n3, n3, kernel_size = 3, padding = 1)
        self.bn5_1 = nn.BatchNorm3d(n3)
        self.bn5_2 = nn.BatchNorm3d(n3)
        """
        # ====================================
        # Upsampling via bilinear upsampling, concatenation (skip connection), followed by 2 conv layers
        # ====================================
        self.deconv2 = nn.Upsample(scale_factor = 2, mode = 'trilinear', align_corners = True)
        self.conv6_1 = nn.Conv3d(n3 + n2, n2, kernel_size = 3, padding = 1)
        self.conv6_2 = nn.Conv3d(n2, n2, kernel_size = 3, padding = 1)
        self.bn6_1 = nn.BatchNorm3d(n2)
        self.bn6_2 = nn.BatchNorm3d(n2)
        
        # ====================================
        # Upsampling via bilinear upsampling, concatenation (skip connection), followed by 2 conv layers
        # ====================================
        self.deconv1 = nn.Upsample(scale_factor = 2, mode = 'trilinear', align_corners = True)
        self.conv7_1 = nn.Conv3d(n2 + n1, n1, kernel_size = 3, padding = 1)
        self.conv7_2 = nn.Conv3d(n1, n1, kernel_size = 3, padding = 1)
        self.bn7_1 = nn.BatchNorm3d(n1)
        self.bn7_2 = nn.BatchNorm3d(n1)
        
        # ====================================
        # Final conv layer - without batch normalization or activation
        # ====================================
        self.conv_pred = nn.Conv3d(n1, num_classes, kernel_size = 3, padding = 1, bias = False)
        
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.max_pool = nn.MaxPool3d(2)
        
    def forward(self, zs):
        #conv1_2, conv2_2, conv3_2, conv4_2 = zs
        conv1_2, conv2_2, conv3_2 = zs
        """
        # ====================================
        # Upsampling via bilinear upsampling, concatenation (skip connection), followed by 2 conv layers
        # ====================================
        deconv3 = self.deconv3(conv4_2)
        skip3 = torch.cat([deconv3, conv3_2], axis = 1)
        conv5_1 = self.relu(self.bn5_1(self.conv5_1(skip3)))
        conv5_2 = self.relu(self.bn5_2(self.conv5_2(conv5_1)))
        """
        # ====================================
        # Upsampling via bilinear upsampling, concatenation (skip connection), followed by 2 conv layers
        # ====================================
        deconv2 = self.deconv2(conv3_2)
        skip2 = torch.cat([deconv2, conv2_2], axis = 1)
        conv6_1 = self.relu(self.bn6_1(self.conv6_1(skip2)))
        conv6_2 = self.relu(self.bn6_2(self.conv6_2(conv6_1)))
        
        # ====================================
        # Upsampling via bilinear upsampling, concatenation (skip connection), followed by 2 conv layers
        # ====================================
        deconv1 = self.deconv1(conv6_2)
        skip1 = torch.cat([deconv1, conv1_2], axis = 1)
        conv7_1 = self.relu(self.bn7_1(self.conv7_1(skip1)))
        conv7_2 = self.relu(self.bn7_2(self.conv7_2(conv7_1)))
        
        # ====================================
        # Final conv layer - without batch normalization or activation
        # ====================================
        pred = self.conv_pred(conv7_2)
        
        return pred


# ====================================
# Discriminator class for Generative Adversarial Network
# ====================================    
class Discriminator(nn.Module):
    def __init__(self, n0 = 16):
        super(Discriminator, self).__init__()
        n1, n2, n3, n4 = 1*n0, 2*n0, 4*n0, 8*n0 # number of channels
        
        # ====================================
        # 1st Conv block - a conv layer
        # ====================================
        self.conv1_1 = nn.Conv2d(1, n0, kernel_size = 4, stride = 2, bias = False)
        
        # ====================================
        # 2nd Conv block - a conv layer followed by batchnorm
        # ====================================
        self.conv2_1 = nn.Conv2d(n0, n1, kernel_size = 4, stride = 2, bias = False)
        self.bn2 = nn.BatchNorm2d(n1)
        
        # ====================================
        # 3rd Conv block - a conv layer followed by batchnorm
        # ====================================
        self.conv3_1 = nn.Conv2d(n1, n2, kernel_size = 4, stride = 2, bias = False)
        self.bn3 = nn.BatchNorm2d(n2)
        
        # ====================================
        # 4th Conv block - a conv layer followed by batchnorm
        # ====================================
        self.conv4_1 = nn.Conv2d(n2, n3, kernel_size = 4, stride = 2, bias = False)
        self.bn4 = nn.BatchNorm2d(n3)
        
        # ====================================
        # 5th Conv block - a conv layer followed by batchnorm
        # ====================================
        self.conv5_1 = nn.Conv2d(n3, n4, kernel_size = 4, stride = 2, bias = False)
        self.bn5 = nn.BatchNorm2d(n4)
        
        # ====================================
        # 5th Conv block - a conv layer followed by batchnorm
        # ====================================
        self.conv6_1 = nn.Conv2d(n4, 1, kernel_size = (2, 6), stride = 1, bias = False)
        
        self.leaky_relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        
        # ====================================
        # 1st Conv block - a conv layer followed by LeakyRelu
        # ====================================
        x = self.conv1_1(x)
        x = self.leaky_relu(x)
        
        # ====================================
        # 2nd Conv block - a conv layer followed batchnorm and LeakyRelu
        # ====================================
        x = self.conv2_1(x)
        x = self.bn2(x)
        x = self.leaky_relu(x)
        
        # ====================================
        # 3rd Conv block - a conv layer followed batchnorm and LeakyRelu
        # ====================================
        x = self.conv3_1(x)
        x = self.bn3(x)
        x = self.leaky_relu(x)
        
        # ====================================
        # 4th Conv block - a conv layer followed batchnorm and LeakyRelu
        # ====================================
        x = self.conv4_1(x)
        x = self.bn4(x)
        x = self.leaky_relu(x)
        
        # ====================================
        # 4th Conv block - a conv layer followed batchnorm and LeakyRelu
        # ====================================
        x = self.conv5_1(x)
        x = self.bn5(x)
        x = self.leaky_relu(x)
        
        # ====================================
        # 6th Conv block - a conv layer followed batchnorm and LeakyRelu
        # ====================================
        x = self.conv6_1(x)
        x = self.sigmoid(x)
        
        return x
    
# ====================================
# Attention class
# ====================================
class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size = 1, stride = 1, padding = 0, bias = True),
            nn.BatchNorm2d(F_int)
            )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size = 1, stride = 1, padding = 0, bias = True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size = 1, stride = 1, padding = 0, bias = True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace = True)
        
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi
    
# ====================================
# Attention Decoder class
# ====================================    
class AttentionDecoder(nn.Module):
    def __init__(self, n0, num_classes):
        super(AttentionDecoder, self).__init__()
        n1, n2, n3, n4 = 1*n0, 2*n0, 4*n0, 8*n0 # number of channels
        
        # ====================================
        # Upsampling via bilinear upsampling, concatenation (skip connection), followed by 2 conv layers
        # ====================================
        self.deconv3 = nn.Upsample(scale_factor = 2, mode = 'bilinear', align_corners = True)
        self.attn5 = AttentionBlock(F_g = n4, F_l = n3, F_int = n3)
        self.conv5_1 = nn.Conv2d(n4 + n3, n3, kernel_size = 3, padding = 1)
        self.conv5_2 = nn.Conv2d(n3, n3, kernel_size = 3, padding = 1)
        self.bn5_1 = nn.BatchNorm2d(n3)
        self.bn5_2 = nn.BatchNorm2d(n3)
        
        # ====================================
        # Upsampling via bilinear upsampling, concatenation (skip connection), followed by 2 conv layers
        # ====================================
        self.deconv2 = nn.Upsample(scale_factor = 2, mode = 'bilinear', align_corners = True)
        self.attn6 = AttentionBlock(F_g = n3, F_l = n2, F_int = n2)
        self.conv6_1 = nn.Conv2d(n3 + n2, n2, kernel_size = 3, padding = 1)
        self.conv6_2 = nn.Conv2d(n2, n2, kernel_size = 3, padding = 1)
        self.bn6_1 = nn.BatchNorm2d(n2)
        self.bn6_2 = nn.BatchNorm2d(n2)
        
        # ====================================
        # Upsampling via bilinear upsampling, concatenation (skip connection), followed by 2 conv layers
        # ====================================
        self.deconv1 = nn.Upsample(scale_factor = 2, mode = 'bilinear', align_corners = True)
        self.attn7 = AttentionBlock(F_g = n2, F_l = n1, F_int = n1)
        self.conv7_1 = nn.Conv2d(n2 + n1, n1, kernel_size = 3, padding = 1)
        self.conv7_2 = nn.Conv2d(n1, n1, kernel_size = 3, padding = 1)
        self.bn7_1 = nn.BatchNorm2d(n1)
        self.bn7_2 = nn.BatchNorm2d(n1)
        
        # ====================================
        # Final conv layer - without batch normalization or activation
        # ====================================
        self.conv_pred = nn.Conv2d(n1, num_classes, kernel_size = 3, padding = 1, bias = False)
        
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.max_pool = nn.MaxPool2d(2)
        
    def forward(self, zs):
        conv1_2, conv2_2, conv3_2, conv4_2 = zs
        
        # ====================================
        # Upsampling via bilinear upsampling, concatenation (skip connection), followed by 2 conv layers
        # ====================================
        deconv3 = self.deconv3(conv4_2)
        attn5 = self.attn5(deconv3, conv3_2)
        skip3 = torch.cat([deconv3, attn5], axis = 1)
        conv5_1 = self.relu(self.bn5_1(self.conv5_1(skip3)))
        conv5_2 = self.relu(self.bn5_2(self.conv5_2(conv5_1)))
        
        # ====================================
        # Upsampling via bilinear upsampling, concatenation (skip connection), followed by 2 conv layers
        # ====================================
        deconv2 = self.deconv2(conv5_2)
        attn6 = self.attn6(deconv2, conv2_2)
        skip2 = torch.cat([deconv2, attn6], axis = 1)
        conv6_1 = self.relu(self.bn6_1(self.conv6_1(skip2)))
        conv6_2 = self.relu(self.bn6_2(self.conv6_2(conv6_1)))
        
        # ====================================
        # Upsampling via bilinear upsampling, concatenation (skip connection), followed by 2 conv layers
        # ====================================
        deconv1 = self.deconv1(conv6_2)
        attn7 = self.attn7(deconv1, conv1_2)
        skip1 = torch.cat([deconv1, attn7], axis = 1)
        conv7_1 = self.relu(self.bn7_1(self.conv7_1(skip1)))
        conv7_2 = self.relu(self.bn7_2(self.conv7_2(conv7_1)))
        
        # ====================================
        # Final conv layer - without batch normalization or activation
        # ====================================
        pred = self.conv_pred(conv7_2)
        
        return pred