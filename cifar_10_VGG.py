import os
import torch
import random
import pathlib
import zipfile
import requests
import torchinfo
import torchvision
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import matplotlib.pyplot as plt
from torch import nn
from PIL import Image
from pathlib import Path
from tqdm import tqdm 
from torchinfo import summary
from typing import Tuple, Dict, List
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

print('Torch version: '+torch.__version__)
print('Torchvision version: '+torchvision.__version__)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("\ndevice: "+device)

DOANLOAD_DATASET = True
LR = 0.001
BATCH_SIZE=512
EPOCH = 10

#############################

#1.torchvision.transforms.Normalize((.5, .5, .5), (.5, .5, .5))裡面的直也許可以做更改

#############################

train_transform = torchvision.transforms.Compose([
   torchvision.transforms.ToTensor(),
   torchvision.transforms.Normalize((.5, .5, .5), (.5, .5, .5)),
   torchvision.transforms.RandomHorizontalFlip(),
   torchvision.transforms.RandomErasing(scale=(0.04,0.2),ratio=(0.5,2)),
   torchvision.transforms.RandomCrop(32, 4),
])

test_transform = torchvision.transforms.Compose([
   torchvision.transforms.ToTensor(),
   torchvision.transforms.Normalize((.5, .5, .5), (.5, .5, .5))
])


##data_path = Path("CIFAR-10-images-master/")
image_path = Path("CIFAR-10-images-master/")
image_path.mkdir(parents = True, exist_ok = True)


# if image_path.is_dir():
#   print(f"\n{image_path} dictionary already exist... skipping download")
# else:
#   print(f"\n{image_path} does not exist, creating one...")
#   image_path.mkdir(parents = True, exist_ok = True)


def walk_through_dir(dir_path):
 """Walks through dir_path. returning it's contents."""
 for dirpath, dirnames, filenames in os.walk(dir_path):
   print(f"There are {len(dirnames)} directions and {len(filenames)} images in {dirpath}")


print(walk_through_dir(image_path))


train_dir = image_path / "train"
test_dir = image_path / "test"
print(f"\ntrain_dir: {train_dir}, \ntest_dir: {test_dir}")


random.seed(42)
#Visualize and image
image_path_list = list(image_path.glob("*/*/*.jpg"))
random_image_path = random.choice(image_path_list)
image_class = random_image_path.parent.stem
img = Image.open(random_image_path)


print(f"\nVisualize image data:")
print(f"Random image path: {random_image_path}")
print(f"Image class: {image_class}")
print(f"Image height: {img.height}")
print(f"image width: {img.width}")


# Turn the image into an array
img_as_array = np.asarray(img)


# Plot the image with matplotlib
plt.figure(figsize = (10, 7))
plt.imshow(img_as_array)
plt.title(f"Image class: {image_class} | Image shape: {img_as_array.shape} -> [height, width, color_channels]")
plt.suptitle(f"Showing a randomly chosen data")
plt.axis(False)
plt.show()


data_transform = transforms.Compose([
   torchvision.transforms.ToTensor(),
   torchvision.transforms.Normalize((.5, .5, .5), (.5, .5, .5)),
   torchvision.transforms.RandomHorizontalFlip(),
   torchvision.transforms.RandomErasing(scale=(0.04,0.2),ratio=(0.5,2)),
   torchvision.transforms.RandomCrop(32, 4),
])


print(f"\nPrint out transformed image data: ")
print(data_transform(img))
print(data_transform(img).shape)
print(f"{data_transform(img).dtype}\n")


def plot_transformed_images(image_paths: list, transform, n=3, seed = None):
 if seed:
   random.seed(seed)
 random_image_paths = random.sample(image_paths, k = n) # k表示樣本長度
 for image_path in random_image_paths:
   with Image.open(image_path) as f:
     fig, ax = plt.subplots(nrows = 1, ncols = 2)
     ax[0].imshow(f)
     ax[0].set_title(f"Origional\nSize: {f.size} ")
     ax[0].axis(False)
    
     transformed_image = transform(f).permute(1, 2, 0) # note we will need to change shape for matplotlib (C, H, W ) -> (H, W, C)
     ax[1].imshow(transformed_image)
     ax[1].set_title(f"Transformed\nShape: {transformed_image.shape}")
     ax[1].axis(False)
    
     fig.suptitle(f"class: {image_path.parent.stem}", fontsize = 16)
     plt.suptitle(f"Comparing origional data and transformed data")
     plt.show()

plot_transformed_images(image_paths = image_path_list,
                       transform = data_transform,
                       n = 3,
                       seed = 42)

print(f"\nLoading image data using ImageFolder \n")
train_data = datasets.ImageFolder(root = train_dir,
                                 transform = data_transform, # a transform for the data
                                 target_transform = None) # a transform for the label/target
test_data = datasets.ImageFolder(root = test_dir,
                                transform = data_transform,)
print(f"\n{train_data}, \n{test_data}\n")
print(f"\n{train_dir}, \n{test_dir}\n")

# Get class names as list
class_names = train_data.classes
print(f"{class_names}\n")
# Get class names as dict
class_dict = train_data.class_to_idx
print(f"{class_dict}\n")
# Check the lengths of our dataset
print(f"Train_data len: {len(train_data)}\nTest_data len: {len(test_data)}\n")
# Index on the train_data Dataset to get a single image and label
img, label = train_data[0][0], train_data[0][1]

print(f"Image tensor:\n {img}")
print(f"Image shape: {img.shape}")
print(f"Image datatype: {img.dtype}")
print(f"Image label: {label}")
print(f"Label datatype: {type(label)}\n")

# rearrange the order dimensions
img_permute = img.permute(1, 2, 0)
print(f"Origional shape: {img.shape} -> [color_channels, height, width]")
print(f"Image permute: {img_permute.shape} ->[height, width, color_channels]\n")

plt.figure(figsize = (10, 7))
plt.imshow(img_permute)
plt.axis(False)
plt.title(class_names[label], fontsize = 14)
plt.suptitle(f"Showing the chosen and transformed picture using ImageFolder")
plt.show()

BATCH_SIZE = 512
NUM_WORKERS = 0
train_dataloader = DataLoader(dataset = train_data,
                             batch_size = BATCH_SIZE,
                             num_workers = NUM_WORKERS,
                             shuffle = True)
test_dataloader = DataLoader(dataset = test_data,
                             batch_size = BATCH_SIZE,
                             num_workers = NUM_WORKERS,
                             shuffle = False)
print(f"Turn loaded images into `DataLoader:\n{train_dataloader},\n{test_dataloader}\n")
print(f"train_dataloader len: {len(train_dataloader)}\ntest_dataloader len: {len(test_dataloader)}")

# Loading Image Data with a Custom `Dataset`
print(f"\nOption 2: Loading Image Data with a Custom Dataset\n")
print(f"train_data.classes: {train_data.classes} \ntrain_data.class_to_idx: {train_data.class_to_idx}\n")

# Creating a helper funcrion to get class names
# Setup path for target directory
target_directory = train_dir
print(f"Target dir: {target_directory}\n")
# Get the class names from the target directory
class_names_found = sorted([entry.name for entry in list(os.scandir(target_directory))])
print(f"class_names_found: {class_names_found}\n")

def find_classes(directory: str) -> Tuple[List[str],Dict[str,int]]:
 """Finds the class folder names in a target directory."""
 # 1. Get the class names by scanning the target directory
 classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
  # 2. Raise an error if class names could not be found
 if not classes:
   raise FileNotFoundError(f"Couldn't find any classes in {directory}... please check file structure")


 # 3. Create a dictionary of index labels (computers prefer numbers rather than strings and labels)
 class_to_idx = {class_name: i for i, class_name in enumerate(classes)}
 return classes, class_to_idx

print(f"find_classes: {find_classes(target_directory)}\n")

class ImageFolderCustom(Dataset):
 # 2. Initialize our custom dataset
 def __init__(self,
              targ_dir: str,
              transform = None):
   # 3. Create class attributes
   # Get all of the image paths
   self.paths = list(pathlib.Path(targ_dir).glob("*/*.jpg"))
   # Setup transforms
   self.transform = transform
   # Create classes and class_to_idx attributes
   self.classes, self.class_to_idx = find_classes(targ_dir)
  # 4. Create a function to load images
 def load_image(self, index: int) ->Image.Image:
   "Opens an image via a path and returns it. "
   image_path = self.paths[index]
   return Image.open(image_path)
 # 5. Overwrite __len__()
 def __len__(self) -> int:
   "Returns the total number of samples. "
   return len(self.paths)
 # Overwrite __getitem__() method to return a particular sample
 def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
   "Returns one sample of data, data and label (X, y)."
   img = self.load_image(index)
   class_name = self.paths[index].parent.name #expects path in format: data_folder/class_name/image.jpg
   class_idx = self.class_to_idx[class_name]

   # Transform of necessary
   if self.transform:
     return self.transform(img), class_idx # return data ,label(X, y)
   else:
     return img, class_idx # return untransformed image and label
  
train_transforms = transforms.Compose([
   torchvision.transforms.ToTensor(),
   torchvision.transforms.Normalize((.5, .5, .5), (.5, .5, .5)),
   torchvision.transforms.RandomHorizontalFlip(),
   torchvision.transforms.RandomErasing(scale=(0.04,0.2),ratio=(0.5,2)),
   torchvision.transforms.RandomCrop(32, 4),
])
test_transforms = transforms.Compose([transforms.ToTensor(),
                                      torchvision.transforms.Normalize((.5, .5, .5), (.5, .5, .5))                             
])


train_data_custom = ImageFolderCustom(targ_dir = train_dir,
                                     transform = train_transforms)
test_data_custom = ImageFolderCustom(targ_dir = test_dir,
                                    transform = test_transforms)


# Check data
print(f"\ntrain_data len: {len(train_data)}\ntrain_data_custom len: {len(train_data_custom)}")
print(f"\ntest_data len: {len(test_data)}\ntest_data_custom len: {len(test_data_custom)}")
print(f"\ntrain_data_custom.classes: {train_data_custom.classes}")
print(f"\ntest_data_custom.class_to_idx: {test_data_custom.class_to_idx}")


# Check for equality between origional ImageFolder Dataset and ImageFolderCustomDataset
print(f"\nCheck for equality between origional ImageFolder Dataset and ImageFolderCustomDataset")
print(train_data_custom.classes == train_data.classes)
print(test_data_custom.classes == test_data.classes)
print(train_data_custom.class_to_idx == train_data.class_to_idx)


# 1. Create a function to take in a dataset
def display_random_images(dataset: torch.utils.data.dataset.Dataset,
                         classes: List[str] = None,
                         n: int = 10,
                         display_shape : bool = True,
                         seed: int = None):
 # 2. Adjust display if n is to high
 if n > 10:
   n = 10
   display_shape = False
   print(f"For display, purpose, n shouldn't be larger than 10, setting and removing shape display.")
  # 3. Set the seed
 if seed:
   random.seed(seed)
 # 4. Get random sample indexes
 random_samples_idx = random.sample(range(len(dataset)),k = n)
 # 5. Setup plot
 plt.figure(figsize = (16, 8))
 # 6. Loop through the random indexes and plot them with matplotlib
 for i, targ_sample in enumerate(random_samples_idx): # 這邊的targ_sample 前面也沒有設過那這參數是哪裡來的？
   targ_image, targ_label = dataset[targ_sample][0], dataset[targ_sample][1]
   # 7. Adjust tensor dimensions for plotting
   targ_image_adjust = targ_image.permute(1, 2, 0) # [color_chennels, height, width] -> [height, width, color_chennels]
   # Plot adjusted samples
   plt.subplot(1, n, i+1)
   plt.imshow(targ_image_adjust)
   plt.axis("off")
   if classes:
     title = f"Class: {classes[targ_label]}"
     if display_shape:
       title = title + f"\n shape: {targ_image_adjust.shape}"
   plt.title(title)
   plt.suptitle(f"Function display random images from train data")
   plt.show()

# Display random images from the ImageFolder created Dataset
display_random_images(train_data,
                     n = 5,
                     classes = class_names,
                     seed = None)

BATCH_SIZE = 512
NUM_WORKERS = 0
#os.cpu_count()
train_dataloader_custom = DataLoader(dataset = train_data_custom,
                                   batch_size = BATCH_SIZE,
                                   num_workers = NUM_WORKERS,
                                   shuffle = True)
test_dataloader_custom = DataLoader(dataset = test_data_custom,
                                   batch_size = BATCH_SIZE,
                                   num_workers = NUM_WORKERS,
                                   shuffle = False)
print(f"\ntrain_dataloader_custom: {train_dataloader_custom}\ntest_dataloader_custom: {test_dataloader_custom}\n")

# Get all image paths
image_path_list = list(image_path.glob("*/*/*.jpg"))
image_path_list [:10]

# Plot random transformed images
plot_transformed_images(
   image_paths = image_path_list,
   transform = train_transforms,
   n = 3,
   seed = None
)


#Creating transforms and loading data for Model 0
simple_transform = transforms.Compose([
   torchvision.transforms.ToTensor(),
   torchvision.transforms.Normalize((.5, .5, .5), (.5, .5, .5)),
   torchvision.transforms.RandomHorizontalFlip(),
   torchvision.transforms.RandomErasing(scale=(0.04,0.2),ratio=(0.5,2)),
   torchvision.transforms.RandomCrop(32, 4),
])
# 1. Load and transform data
train_data_simple = datasets.ImageFolder(root = train_dir,
                                        transform = simple_transform)
test_data_simple = datasets.ImageFolder(root = test_dir,
                                       transform = simple_transform)


# 2. Turn the datasets into DataLoaders
Batch_size = 512
NUM_WORKERS = 0
# Create DataLoader's
train_dataloader_simple = DataLoader(dataset = train_data_simple,
                                    batch_size = BATCH_SIZE,
                                    shuffle = True,
                                    num_workers = NUM_WORKERS)
test_dataloader_simple = DataLoader(dataset = test_data_simple,
                                    batch_size = BATCH_SIZE,
                                    shuffle = False,
                                    num_workers = NUM_WORKERS)

####################################

##1. 有沒有可能是沒有softmax()
##2. 最好使用预训练的权重  大多数流行的backone比如resnet都有再imagenet数据集上与训练过，那么使用这种权重，比起随即重新训练，显然要可靠不少注意调整学习率。

####################################

class VGG16(nn.Module):
    """
    dropput()
    而在全連結層之後會接一個 dropout 函數，而它是為了來避免 overfitting 的神器，通常在訓練過程會使用（這裡的 p = 0.5）
    意思就是會這些神經元會隨機的被關掉，要這樣的做的原因是避免神經網路在訓練的時候防止特徵之間有合作的關係．
    隨機的關掉一些節點後，原本的神經網路就被逼迫著從剩下不完整的網路來學習，而不是每次都透過特定神經元的特徵來分類
    ReLu()/Sigmoid()
    在類神經網路中使用激勵函數，主要是利用非線性方程式，解決非線性問題，若不使用激勵函數，類神經網路即是以線性的方式組合運算，
    因為隱藏層以及輸出層皆是將上層之結果輸入，並以線性組合計算，作為這一層的輸出，使得輸出與輸入只存在著線性關係，而現實中，
    所有問題皆屬於非線性問題，因此，若無使用非線性之激勵函數，則類神經網路訓練出之模型便失去意義。
    layer解說
    https://medium.com/jameslearningnote/資料分析-機器學習-第5-1講-卷積神經網絡介紹-convolutional-neural-network-4f8249d65d4f
    MaxPool()
    做平坦化
    """
    def __init__(self):
        super(VGG16, self).__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 128, 3, padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2)
        )
        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2)
        )
        self.conv_block_4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, padding = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, padding = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2)
        )
        self.conv_block_5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, padding = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, padding = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.AvgPool2d(1, 1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 10),
         )
    def forward(self, x):
            x = self.conv_block_1(x)
            x = self.conv_block_2(x)
            x = self.conv_block_3(x)
            x = self.conv_block_4(x)
            x = self.conv_block_5(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return x
model_0 = VGG16()
print(f"model_0 structure: {model_0}\n")

#Try a forward pass on a single image(to test the model)
# Get a single image
image_batch, label_batch = next(iter(train_dataloader_simple))
image_batch.shape, label_batch.shape
# Try a forward pass
print(f"Do a forward pass with model_0: {model_0(image_batch.to(device))}")
# Install torchinfo, import if it's available
print(f"Print out data using torchinfo summary: \n{summary(model_0, input_size = [1, 3, 32 ,32])}\n")

# 7.5 Create train and test loops functions
# Create a train_step()
def train_step(model : torch.nn.Module,
              dataloader : torch.utils.data.DataLoader,
              loss_fn : torch.nn.Module,
              optimizer : torch.optim.Optimizer,
              device : device):
 # Put the model in train mode
 model.train()

 # Setup train loss and train accuracy values
 train_loss, train_acc = 0, 0

 # Loop through data loader data batches
 for batch, (X,y) in enumerate(dataloader):
   # Send data to the target device
   X, y = X.to(device), y.to(device)

   # 1. Forward pass
   y_pred = model(X) # output model logits

   # 2. Cavulate the loss
   loss = loss_fn(y_pred, y)
   train_loss += loss.item()

   # 3. Optimizer zero grad
   optimizer.zero_grad()

   # 4. Loss backward 沒有定義 還有一些像是.sum() .item() .to()
   loss.backward()

   # 5. Optimizer step 與網路上定義的不太一樣 網路上的多了一些group這對訓練有影響嗎
   optimizer.step()

   # Caculate accuracy metric
   y_pred_class = torch.argmax(torch.softmax(y_pred, dim = 1), dim = 1)
   train_acc += (y_pred_class == y).sum().item()/len(y_pred)

 # Adjust metrics to get average loss and accuracy per batch
 train_loss = train_loss / len(dataloader)
 train_acc = train_acc / len(dataloader)
 return train_loss, train_acc


# Create a test step
def test_step(model : torch.nn.Module,
             dataloader : torch.utils.data.DataLoader,
             loss_fn : torch.nn.Module,
             device : device):
 # Put model in eval mode
 model.eval()

 # Setup test loss and test accuracy values
 test_loss, test_acc = 0, 0

 # Turn on inference_mode
 with torch.inference_mode():
   # Loop through DataLoader batches
   for batch, (X, y) in enumerate(dataloader):
     # Send data to the target device
     X, y = X.to(device), y.to(device)

     # 1. Forward pass
     test_pred_logits = model(X)
    
     # 2. Caculate the loss
     loss = loss_fn(test_pred_logits, y)
     test_loss += loss.item()
    
     # Caculate the accurcy
     test_pred_labels = test_pred_logits.argmax(dim = 1)
     test_acc += ((test_pred_labels == y).sum().item() / len(test_pred_labels))


 # Adjust metrics to get average loss and accuracy per batch
 test_loss = test_loss / len(dataloader)
 test_acc = test_acc / len(dataloader)


 return test_loss, test_acc


def train(model : torch.nn.Module,
         train_dataloader : torch.utils.data.DataLoader,
         test_dataloader : torch.utils.data.DataLoader,
         optimizer : torch.optim.Optimizer,
         loss_fn : torch.nn.Module = nn.CrossEntropyLoss(),
         epochs : int = 5,):
 # Create empty results dictionary
 results = {"train_loss": [],
            "train_acc": [],
            "test_loss": [],
            "test_acc": []}
  # Loop through training and testing steps for a number of epochs
 for epoch in tqdm(range(epochs)):
   train_loss, train_acc = train_step(model = model,
                                      dataloader = train_dataloader,
                                      loss_fn = loss_fn,
                                      optimizer = optimizer,
                                      device = device)
   test_loss, test_acc = test_step(model = model,
                                   dataloader = test_dataloader,
                                   loss_fn = loss_fn,
                                   device = device)
   # 4. Print out what's heppening
   print(f"Epoch: {epoch} | Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f} | Test acc: {test_acc:.4f}")


   # 5. Update results dictionary
   results["train_loss"].append(train_loss)
   results["train_acc"].append(train_acc)
   results["test_loss"].append(test_loss)
   results["test_acc"].append(test_acc)


 # 6.Return the filled results at the end of the epochs
 return results

### 7.7 Train and evaluate model 0

# Set random seeds
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# Set number of epochs
NUM_EPOCHS = 4

# Recreate an instance of TinyVGG
model_0 = VGG16()

# Setup loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params = model_0.parameters(),
                            lr = LR,
                            momentum=0.9,
                            weight_decay=5e-3
                            )
# Srart the timer
from timeit import default_timer as timer
start_time = timer()

# Train model_0
model_0_results = train(model = model_0,
                       train_dataloader = train_dataloader_simple,
                       test_dataloader = test_dataloader_simple,
                       optimizer = optimizer,
                       loss_fn = loss_fn,
                       epochs = NUM_EPOCHS)

# End the timer and print out how long it took
end_time = timer()
print(f"Total trianing time: {end_time-start_time:.3f} seconds")


def plot_loss_curves(results: Dict[str, List[float]]):
 """Plots training curve of a results dictionary."""
 # Get the loss values of results dictionary(training and test)
 loss = results["train_loss"]
 test_loss = results["test_loss"]


 # Get the accuracy values of the results dictionary (training and testing)
 accuracy = results ["train_acc"]
 test_accuracy = results["test_acc"]


 # Figure out how many wpochsthere were
 epochs = range(len(results["train_loss"]))


 # Setup a plot
 plt.figure(figsize = (15, 7))


 # Plot the loss
 plt.subplot(1, 2, 1)
 plt.plot(epochs, loss, label="train_loss")
 plt.plot(epochs, test_loss, label = "test_loss")
 plt.title("Loss")
 plt.xlabel("Epochs")
 plt.legend()
 plt.show()


 # Plot the accuracy
 plt.subplot(1, 2, 2)
 plt.plot(epochs, accuracy, label="train accuracy")
 plt.plot(epochs, test_accuracy, label= "test_accuracy")
 plt.title("Accuracy")
 plt.legend();
 plt.show()


plot_loss_curves(model_0_results)



