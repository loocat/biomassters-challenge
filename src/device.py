import sys
import torch
import torchvision

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    print('torch version = ', torch.__version__)
    print('torchvision version = ', torchvision.__version__)

    print('CUDA available: ' + str(torch.cuda.is_available()))
    print('cuDNN version: ' + str(torch.backends.cudnn.version()))
    a = torch.cuda.FloatTensor(2).zero_()
    print('Tensor a = ' + str(a))
    b = torch.randn(2).cuda()
    print('Tensor b = ' + str(b))
    c = a + b
    print('Tensor c = ' + str(c))

    sys.exit(0)