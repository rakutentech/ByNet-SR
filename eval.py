import argparse
import torch
from torch.autograd import Variable
import numpy as np
import time, math, glob
import scipy.io as sio

parser = argparse.ArgumentParser(description="PyTorch ByNet Eval")
parser.add_argument("--cuda", action="store_true", help="use cuda?")
parser.add_argument("--model", default="model/model_epoch_40.pth", type=str, help="Model path, Default=model/model_epoch_40.pth")
parser.add_argument("--dataset", default="Set5", type=str, help="Dataset name, Default: Set5")

def PSNR(pred, gt, shave_border=0):
    height, width = pred.shape[:2]
    pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
    gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)

opt = parser.parse_args()
cuda = opt.cuda

if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

model = torch.load(opt.model)["model"].module

scales = [2,3,4]

image_list = glob.glob(opt.dataset+"_mat/*.*") 

for scale in scales:
    avg_psnr_predicted = 0.0
    avg_psnr_bicubic = 0.0
    avg_elapsed_time = 0.0
    count = 0.0
    for image_name in image_list:
        if "_x" + str(scale) in image_name:
            count += 1
            print("Processing ", image_name)
            im_gt_y = sio.loadmat(image_name)["im_gt_y"]
            im_b_y = sio.loadmat(image_name)["im_b_y"]

            im_gt_y = im_gt_y.astype(float)
            im_b_y = im_b_y.astype(float)      

            psnr_bicubic = PSNR(im_gt_y, im_b_y,shave_border=scale)
            avg_psnr_bicubic += psnr_bicubic

            im_input = im_b_y/255.

            im_input = Variable(torch.from_numpy(im_input).float()).view(1, -1, im_input.shape[0], im_input.shape[1])

            if cuda:
                model = model.cuda()
                im_input = im_input.cuda()
            else:
                model = model.cpu()
                
            start_time = time.time()
            HR = model(im_input)
            elapsed_time = time.time() - start_time
            avg_elapsed_time += elapsed_time

            HR = HR.cpu()

            im_h_y = HR.data[0].numpy().astype(np.float32)

            im_h_y = im_h_y*255.
            im_h_y[im_h_y<0] = 0
            im_h_y[im_h_y>255.] = 255.
            im_h_y = im_h_y[0,:,:]

            psnr_predicted = PSNR(im_gt_y, im_h_y,shave_border=scale)
            avg_psnr_predicted += psnr_predicted

    print("Scale=", scale)
    print("Dataset=", opt.dataset)
    print("PSNR_predicted=", avg_psnr_predicted/count)
    print("PSNR_bicubic=", avg_psnr_bicubic/count)
    print("It takes average {}s for processing".format(avg_elapsed_time/count))
