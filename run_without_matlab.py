import torch, os, argparse, glob, time
from torch.autograd import Variable
import numpy as np
from PIL import Image
import scipy.misc
import cv2

parser = argparse.ArgumentParser(description="PyTorch ByNet Enhance (without matlab)")
parser.add_argument("--cuda", action="store_true", help="use cuda?")
parser.add_argument("--denoising", action="store_true", help="use opencv denoising function?")
parser.add_argument("--model", default="model/model_epoch_40.pth", type=str, help="Model path, Default=model/model_epoch_40.pth")
parser.add_argument("--folder", type=str, help="Folder name")

def colorize(y, ycrcb):
    img = np.zeros((y.shape[0], y.shape[1], 3), np.uint8)
    img[:,:,0] = y
    img[:,:,1] = ycrcb[:,:,1]
    img[:,:,2] = ycrcb[:,:,2]
    return img

opt = parser.parse_args()
cuda = opt.cuda

if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

model = torch.load(opt.model)["model"].module

image_list = glob.glob(os.getcwd()+ "/" + opt.folder+"/*.*") 

for image_name in image_list:
    print("Processing ", image_name)

    image = Image.open(image_name).convert("RGB")
    image.thumbnail((800, 800), Image.ANTIALIAS)

    ycrcb_l = np.array(image.convert("YCbCr"))

    im_input = ycrcb_l[:,:,0] / 255.

    im_input = Variable(torch.from_numpy(im_input).float()).view(1, -1, im_input.shape[0], im_input.shape[1]) 

    if cuda:
        model = model.cuda()
        im_input = im_input.cuda()
    else:
        model = model.cpu()
       
    start_time = time.time()
    out = model(im_input)
    elapsed_time = time.time() - start_time

    out = out.cpu()
    im_h_y = out.data[0].numpy().astype(np.float32)

    # Predict results
    im_h_y = im_h_y*255.
    im_h_y[im_h_y<0] = 0
    im_h_y[im_h_y>255.] = 255.
    im_h_y = im_h_y[0,:,:]

    im_pred = colorize(im_h_y, ycrcb_l)
    im_pred = Image.fromarray(im_pred, "YCbCr").convert("RGB")

    if opt.denoising:
        cv_img = np.array(im_pred.convert("RGB"))[:, :, ::-1]
        cv_img = cv2.fastNlMeansDenoisingColored(cv_img,None,5,5,3,10)
        cv_img = cv2.cvtColor(cv_img,cv2.COLOR_BGR2RGB)
        im_pred = Image.fromarray(cv_img)

    folder = "result/" + image_name.split("/")[-2]
    if not os.path.exists(folder):
        os.makedirs(folder)
    scipy.misc.imsave(folder + "/" + image_name.split('/')[-1], im_pred)
    print("It takes average {}s for processing".format(elapsed_time))
