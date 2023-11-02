import os
import cv2
import torch
import argparse
from torch.nn import functional as F
import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description='STVSR for a pair of images')
parser.add_argument('--img', dest='img', nargs=2, required=True)
parser.add_argument('--exp', default=2, type=int)
parser.add_argument('--ratio', default=0, type=float, help='inference ratio between two images with 0 - 1 range')
parser.add_argument('--model', dest='modelDir', type=str, default='train_log', help='directory with trained model files')

args = parser.parse_args()

from train_log.model import Model
model = Model()
model.device()
model.load_model('train_log')
model.eval()

if args.img[0].endswith('.exr') and args.img[1].endswith('.exr'):
    img0 = cv2.imread(args.img[0], cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH)
    img1 = cv2.imread(args.img[1], cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH)
    img0 = cv2.resize(img0, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    img1 = cv2.resize(img1, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    img0 = (torch.tensor(img0.transpose(2, 0, 1)).to(device)).unsqueeze(0)
    img1 = (torch.tensor(img1.transpose(2, 0, 1)).to(device)).unsqueeze(0)    
else:
    img0 = cv2.imread(args.img[0], cv2.IMREAD_UNCHANGED)
    img1 = cv2.imread(args.img[1], cv2.IMREAD_UNCHANGED)
    img0 = cv2.resize(img0, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    img1 = cv2.resize(img1, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    img0 = (torch.tensor(img0.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)
    img1 = (torch.tensor(img1.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)
    
n, c, h, w = img0.shape
ph = ((h - 1) // 32 + 1) * 32
pw = ((w - 1) // 32 + 1) * 32
padding = (0, pw - w, 0, ph - h)
img0 = F.pad(img0, padding)
img1 = F.pad(img1, padding)

if args.ratio:
    print('ratio={}'.format(args.ratio))
    img_list = model.inference(img0, img1, timestep=args.ratio)
else:
    n = 2 ** args.exp - 1    
    time_list = [0]
    for i in range(n):
        time_list.append((i+1) * 1. / (n+1))
    time_list.append(1)
    print(time_list)
    img_list = model.inference(img0, img1, timestep=time_list)
    
if not os.path.exists('output'):
    os.mkdir('output')
for i in range(len(img_list)):
    if args.img[0].endswith('.exr') and args.img[1].endswith('.exr'):
        cv2.imwrite('output/img{}.exr'.format(i), (img_list[i][0]).cpu().numpy().transpose(1, 2, 0)[:h, :w], [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_HALF])
    else:
        cv2.imwrite('output/img{}.png'.format(i), (img_list[i][0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w])
