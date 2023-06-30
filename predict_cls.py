import torch
from torchvision import transforms
import os
import numpy as np
from natsort import natsorted
from tqdm.auto import tqdm
from PIL import Image
from mobilenet import CustomMobileNetv2
import argparse

def get_args():
    """
    read the input arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--inPath', type=str, default='', 
                        help='Input directory of images to classify')
    args = parser.parse_args()
    return args

args = get_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

crop_size = 224

test_transform = transforms.Compose([
    transforms.Resize(crop_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

model = CustomMobileNetv2(3).to(device)
model.load_state_dict(torch.load('cls_model/weights_best.pth'))
model.eval()

inPath = args.inPath
predictions = []
with torch.no_grad():
    for i in tqdm(natsorted(os.listdir(inPath))):
        img_pth = os.path.join(inPath, i)
        img = Image.open(img_pth)
        img = test_transform(img)
        img = img.unsqueeze(0).to(device)
        output = model(img)
        preds = output.argmax(1)
        if preds.cpu().detach().numpy()[0] == 0:
            predictions.append('adding')
        if preds.cpu().detach().numpy()[0] == 1:
            predictions.append('none')
        if preds.cpu().detach().numpy()[0] == 2:
            predictions.append('stirring')

predictions = np.array(predictions)
np.savetxt('{}.txt'.format(inPath), predictions, fmt='%s')
