import os
import shutil
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import concurrent.futures
from torch.utils.data import Dataset, DataLoader

# Utility function to create directory if not exists
def create_directory(directories):
    for dir in directories:
        if not os.path.exists(dir): os.makedirs(dir)

# Process and split dataset
def process_and_split_dataset(rawdata_dir, output_dir = "./dataset/", extensions='.jpg'):
    create_directory([rawdata_dir, output_dir])

    # Filter images with specified extensions
    images = [f for f in os.listdir(rawdata_dir) if f.endswith(extensions)]
    
    # Split the images into sets
    train_images, temp_images = train_test_split(images, train_size=0.8)
    val_images, test_images = train_test_split(temp_images, train_size=0.5)

    # Define function for saving and grayscaling images
    def process_images(images, set_name):
        cr_dir = os.path.join(output_dir, set_name, "cr_imgs")
        bw_dir = os.path.join(output_dir, set_name, "bw_imgs")
        create_directory([cr_dir, bw_dir])

        def process_image(image):
            # Copy to directory
            shutil.copy(os.path.join(rawdata_dir, image), os.path.join(cr_dir, image))
            img = Image.open(os.path.join(rawdata_dir, image)).convert("L")
            img.save(os.path.join(bw_dir, image))

        with concurrent.futures.ThreadPoolExecutor() as executor:
            list(tqdm(executor.map(process_image, images), total=len(images), desc=f"Processing {set_name} set"))

    # Process each set
    process_images(train_images, 'train')
    process_images(val_images, 'val')
    process_images(test_images, 'test')

# Dataset and DataLoader modifications for GAN
class ColorizationDataset(Dataset):
    def __init__(self, bw_img_dir, cr_img_dir, max_dataset_size, transform=None,):
        self.bw_img_dir = bw_img_dir
        self.cr_img_dir = cr_img_dir
        self.transform = transform

        self.image_files = [f for f in os.listdir(bw_img_dir) if f.endswith('.jpg') or f.endswith('.png')]
        self.image_files = self.image_files[:max_dataset_size]  # Limit the dataset size

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        bw_img_path = os.path.join(self.bw_img_dir, img_name)
        bw_img = Image.open(bw_img_path).convert('L')  # Ensure image is grayscale

        cr_img_path = os.path.join(self.cr_img_dir, img_name)
        cr_img = Image.open(cr_img_path).convert('RGB')  # Ensure image is RGB

        if self.transform:
            bw_img = self.transform(bw_img)
            cr_img = self.transform(cr_img)

        return bw_img, cr_img
    
def load_data(transform, batch_size, max_dataset_size):
    # Creating dataset and data loader instances
    train_dataset = ColorizationDataset("./dataset/train/bw_imgs", "./dataset/train/cr_imgs", max_dataset_size, transform=transform)
    val_dataset = ColorizationDataset("./dataset/val/bw_imgs", "./dataset/val/cr_imgs", max_dataset_size, transform=transform)
    test_dataset = ColorizationDataset("./dataset/test/bw_imgs", "./dataset/test/cr_imgs", max_dataset_size, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, val_loader, test_loader