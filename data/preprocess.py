import cv2
import numpy as np
import os
import glob
from tqdm import tqdm

def crop_image_from_gray(img, tol=7):
    # crop black borders
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1),mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol
        
        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
        if (check_shape == 0): 
            return img 
        else:
            img1 = img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            img2 = img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            img3 = img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
            img = np.stack([img1, img2, img3], axis=-1)
        return img

def ben_graham_preprocessing(image_path, target_size=(512, 512)):
    # load rgb
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # crop
    img = crop_image_from_gray(img)
    
    # resize
    img = cv2.resize(img, target_size)
    
    # subtract local avg
    sigmaX = target_size[0] / 10
    img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0,0), sigmaX), -4, 128)
    
    return img

def process_dataset(input_dir, output_dir, target_size=(512, 512)):
    # create output dir
    os.makedirs(output_dir, exist_ok=True)
    image_paths = glob.glob(os.path.join(input_dir, "*.png"))
    
    print(f"Processing {len(image_paths)} images...")
    for path in tqdm(image_paths):
        filename = os.path.basename(path)
        save_path = os.path.join(output_dir, filename)
        
        processed_img = ben_graham_preprocessing(path, target_size)
        
        # convert bgr to save
        processed_img = cv2.cvtColor(processed_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, processed_img)

if __name__ == "__main__":
    # define paths
    raw_images_dir = os.path.join(os.path.dirname(__file__), "raw", "train_images")
    processed_images_dir = os.path.join(os.path.dirname(__file__), "processed", "train_images_512")
    
    # run
    if os.path.exists(raw_images_dir):
        process_dataset(raw_images_dir, processed_images_dir)
    else:
        print("Raw images not found.")