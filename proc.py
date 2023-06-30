import os
import cv2
import argparse
import json
from tqdm import tqdm
import numpy as np
from natsort import natsorted

def get_args():
    '''
    read the input arguments.
    '''
    parser = argparse.ArgumentParser()

    parser.add_argument('--inPath', type=str, default=None, 
                        help='Input the path to the directory containing the images')
    parser.add_argument('--outPath', type=str, default=None, 
                        help='Enter the output directory where the shaded pixel outputs are saved frame wise')
    parser.add_argument('--inClsFile', type=str, default=None,
                        help='Enter the filename with action class predictions')
    parser.add_argument('--outVidName', type=str, default='output.mp4',
                        help='Enter the name of the video to be rendered from the shaded images')
    args = parser.parse_args()
    print(json.dumps(vars(args), indent=4))
    return args

def denoise(mask):
    '''
    Predicted segmented masks tend to have a lot of small blots (where the hand is not present)
    to reduce this noise and to make the masks a bit more coherent and smooth, the following 
    morphological operations are performed
    Erosion - 2 iters (removes all the blots which are small)
    Dialation - 2 iters (Dialates the eroded mask to previos state)
    Applying this instead of opening and closing operations for more finegrained control over denoising
    '''
    mask = mask.astype('float32')
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    mask = cv2.erode(mask,kernel,iterations = 2)
    mask = cv2.dilate(mask, kernel, iterations= 2)
    return mask

def crawl_dir(path):
    '''
    Crawl the predictions directory and create two lists containing
    the relative paths of images and their corresponding masks
    masks have the same name as the images except with a .png extension
    '''
    pathsList = natsorted(os.listdir(path))
    mskPthLst = [os.path.join(path, i) for i in pathsList if i.split('.')[-1] == 'png']
    imgPthLst = [os.path.join(path, i) for i in pathsList if i.split('.')[-1] == 'jpg']
    return imgPthLst, mskPthLst

def overlay_action(inpath, infile):
    print('-----------overlaying action on preshaded images-----------')
    act_preds = np.genfromtxt(infile, dtype='str')
    for imgfile, action in tqdm(zip(natsorted(os.listdir(inpath)), act_preds), total=len(act_preds)):
        img = cv2.imread(os.path.join(inpath, imgfile))
        cv2.putText(img, action, (10,50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (209, 80, 0, 255), 3)
        cv2.imwrite(os.path.join(inpath, imgfile), img)

def overlay_masks(inpath, outpath):
    print('-----------overlaying masks on images-----------')
    # Crawl the path to get the list of relative images and masks path 
    imgPthLst, mskPthLst = crawl_dir(inpath)
    # Loop over the images and masks pair and output them in the outDir
    for img, msk in tqdm(zip(imgPthLst, mskPthLst), total=len(imgPthLst)):
        # Extract the image name and create the output path for shaded images
        outImgName = img.split('/')[-1]+'.jpg'
        outdir = os.path.join(outpath, outImgName)
        # reading the image and creating a copy
        imgArr = cv2.imread(img)
        greenHand = np.copy(imgArr)
        h, w, _ = imgArr.shape
        # denoising the mask
        mskArr = denoise(cv2.imread(msk))
        # reshaping it to the parent image size
        mskArr = cv2.resize(src=mskArr,dsize=(w,h))
        # # Binary thresholding the mask
        _, mskArr = cv2.threshold(mskArr, thresh=180, maxval=255, type=cv2.THRESH_BINARY)
        # # Boolean indexing and assignment based on mask
        greenHand[(mskArr==255).all(-1)] = [0,255,0]
        greenHand = cv2.addWeighted(greenHand, 0.3, imgArr, 0.7, 0, greenHand)
        cv2.imwrite(img=greenHand, filename=outdir)

if __name__ == '__main__':
    # Read and parse input args
    args = get_args()
    inPath = args.inPath
    outPath = args.outPath
    inClsFile = args.inClsFile
    outVidName = args.outVidName

    # create the outPath if it doesn't exist
    if not os.path.exists(outPath):
        os.mkdir(outPath)

    # Overlay the masks and save the outputs to a directory
    overlay_masks(inPath, outPath)
    overlay_action(outPath, inClsFile)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    video = cv2.VideoWriter(outVidName, fourcc, 24, (1920, 1080))
    # crawl the overlay dir and create a list of relative paths
    overlayPaths = [os.path.join(outPath, i) for i in natsorted(os.listdir(outPath))]
    print('-----------Creating Video-----------')
    for i in tqdm(overlayPaths, total=len(overlayPaths)):
        img = cv2.imread(i)
        video.write(img)

    cv2.destroyAllWindows()
    video.release()