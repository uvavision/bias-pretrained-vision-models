from PIL import Image
from torchvision import transforms 
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import os
import tqdm
import random 
import argparse
from multiprocessing import pool
from multiprocessing.dummy import Pool as ThreadPool

def main(path, save_path, split_start, split_end):
    preprocess = transforms.Compose([
            Resize(256, interpolation=Image.BICUBIC),
    ])
    openimages_imgs = os.listdir(path)
    #openimages_computed = set(os.listdir(save_path))
    for index, img in enumerate(tqdm.tqdm(openimages_imgs[split_start:split_end])):
        #if img not in openimages_computed:
        im = Image.open(path+img)
        img_transformed = preprocess(im)
        img_transformed.save(save_path+img)  
        
preprocess = transforms.Compose([
        Resize(256, interpolation=Image.BICUBIC),
])        
def resize_img(img):
    im = Image.open("../VisionResearch/finetuneClip/openimages_dataset/train/"+img)
    img_transformed = preprocess(im)
    img_transformed.save("/data/openimages/train/"+img)      
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split_start', type=int)
    parser.add_argument('--split_end', type=int)    
    args = parser.parse_args()
    
    openimages_imgs = os.listdir("../VisionResearch/finetuneClip/openimages_dataset/train/")
    imagesList = openimages_imgs[args.split_start:args.split_end]
    # Create thread pool
    pool = ThreadPool(10)
    pool.map(resize_img, imagesList)


    #main("../VisionResearch/finetuneClip/openimages_dataset/val/", "/data/openimages/val/")
    #main("../VisionResearch/finetuneClip/openimages_dataset/train/", "/data/openimages/train/", args.split_start, args.split_end)
