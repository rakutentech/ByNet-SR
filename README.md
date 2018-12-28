# PyTorch ByNet
Implementation of ICIP2017 Paper: ["ByNet-SR: Image Super Resolution with a Bypass Connection Network
"](http://bjornstenger.github.io/papers/xu_icip2017.pdf) in PyTorch

### Citation

If you find the code and datasets useful in your research, please cite:
    
    @InProceedings{ByNet,
        author    = {Xu, J. and Chae, Y. and Stenger, B.}, 
        title     = {{ByNet-SR}: Image Super Resolution with a Bypass Connection Network}, 
        booktitle = {IEEE International Conference on Image Processing},
        year      = {2017}
    }

## Usage
### Training
```
usage: main_bynet.py [-h] [--batchSize BATCHSIZE] [--nEpochs NEPOCHS] [--lr LR]
               [--step STEP] [--cuda] [--resume RESUME]
               [--start-epoch START_EPOCH] [--clip CLIP] [--threads THREADS]
               [--momentum MOMENTUM] [--weight-decay WEIGHT_DECAY]
               [--pretrained PRETRAINED]

optional arguments:
  -h, --help            show this help message and exit
  --batchSize BATCHSIZE
                        Training batch size
  --nEpochs NEPOCHS     Number of epochs to train
  --lr LR               Init learning Rate. Default=0.1
  --step STEP           Sets the learning rate to the initial LR decayed by
                        momentum every n epochs, Default: n=10
  --cuda                use cuda?
  --resume RESUME       Path to latest checkpoint (default: none)
  --start-epoch START_EPOCH
                        Manual epoch number (useful on restarts)
  --clip CLIP           Clipping Gradients. Default=0.5
  --threads THREADS     Number of threads for data loader to use
  --momentum MOMENTUM   Momentum
  --weight-decay WEIGHT_DECAY, --wd WEIGHT_DECAY
                        weight decay, Default: 1e-4
```
An example of training usage is shown as follows:
```
python main_bynet.py --cuda
```

### Evaluation
```
usage: eval.py [-h] [--cuda] [--model MODEL] [--dataset DATASET]
               [--scale SCALE]

PyTorch ByNet Eval

optional arguments:
  -h, --help         Show this help message and exit
  --cuda             use cuda?
  --model MODEL      Model path. Default=model/model_epoch_40.pth
  --dataset DATASET  Dataset name, Default: Set5
```
An example of training usage is shown as follows:
```
python eval.py --cuda
```

### Demo
```
usage: demo.py [-h] [--cuda] [--model MODEL] [--image IMAGE] [--scale SCALE]
               
optional arguments:
  -h, --help            Show this help message and exit
  --cuda                Use cuda
  --model               Model path. Default=model/model_epoch_40.pth
  --image               Image name. Default=butterfly_GT
  --scale               Scale factor, Default: 4
```
An example of demo usage is shown as follows:
```
python demo.py --cuda
```

### Image Process without Matlab
```
usage: run_without_matlab.py [-h] [--cuda] [--model MODEL] [--folder FOLDER]

PyTorch ByNet Enhance (without matlab)

optional arguments:
  -h, --help       show this help message and exit
  --cuda           use cuda?
  --model MODEL    Model path, Default=model/model_epoch_40.pth
  --folder FOLDER  Folder name
```
An example of demo usage is shown as follows:
```
python run_without_matlab.py --cuda --folder Set5
```
Put all images in the folder you defined and the results will be appeared in result folder

### Prepare Training dataset
  - We provide a simple hdf5 format training sample in data folder with 'data' and 'label' keys, the training data is generated with Matlab Bicubic Interplotation, please refer [Code for Data Generation](https://github.com/twtygqyy/pytorch-vdsr/tree/master/data) for creating training files.

### Performance
  - We provide a pretrained ByNet9 model trained on [291](https://drive.google.com/open?id=1Rt3asDLuMgLuJvPA1YrhyjWhb97Ly742) images with data augmentation
  - No bias is used in this implementation
  - Performance in PSNR on Set5
  
| Scale        | PNSR          |
| ------------- |:-------------:| 
| 2x      | 37.75      | 
| 3x      | 33.96      | 
| 4x      | 31.60      | 

### Result
From left to right are ground-truth image, bicubic and ByNet
<p>
  <img src='Set5/butterfly_GT.bmp' height='200' width='200'/>
  <img src='result/input.bmp' height='200' width='200'/>
  <img src='result/output.bmp' height='200' width='200'/>
</p>
