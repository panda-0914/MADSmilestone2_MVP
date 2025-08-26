"""ML models and data definitions utilized within the Milestone2 project."""
import torch.nn as nn
import torch
import torchvision
import torchvision.transforms as transforms
import os
from PIL import Image
from skimage.io import imread
import numpy as np
import torch.nn.functional as F


class ResizedClocks:
    """Resized clock drawing dataset."""

    def __init__(self, round, round_labels, data_path=None, normalize_=None):
        """Define the dataset.

        Args:
            round (int): Round to grab images from.
            round_labels (list of tuples): Corresponding values for the round.
            data_path (str): Path to the data directory. If None, uses default Google Drive path.
            normalize_ (bool): Whether to apply normalization transforms.
        """
        self.round = round
        self.vals = round_labels
        
        # Set default path if not provided
        if data_path is None:
            self.data_path = '/content/gdrive/MyDrive/Data/Nhats Dataset/NHATS_R11_ClockDrawings_V2'
        else:
            self.data_path = data_path
            
        if normalize_ == True:
            processes = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
            rgb_val = True
        else:
            processes = transforms.ToTensor()
            rgb_val = None
        self.transform = processes
        self.rgb = rgb_val

    def __len__(self):
        """Define dataset length."""
        return len(self.vals)

    def __getitem__(self, idx):
        """Loops through indexed items in dataset."""
        spid = self.vals[idx][0]
        label = torch.tensor(int(self.vals[idx][1]))
        
        # Construct file path
        filename = f"{spid}.tif"
        file_path = os.path.join(self.data_path, filename)
        
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                print(f"Warning: File {file_path} does not exist")
                return None
                
            # Load image directly from file system
            im = Image.open(file_path)

            if self.rgb == True:
                # Convert to RGB for color processing
                gray = im.convert("RGB")
            else:
                # Convert to binary (1-bit pixels, black and white)
                gray = im.convert("1")

            # Resize image
            resized = gray.resize((284, 368))
            im_arr = np.array(resized)

            if self.transform:
                im_arr = self.transform(im_arr)

            return im_arr, label

        except Exception as e:
            print(f"Error loading image {file_path}: {str(e)}")
            return None


# original size: 2560, 3312
class ConvNet(nn.Module):
    """From scratch CNN to label dementia."""

    def __init__(self):
        """Define CNN."""
        super(ConvNet, self).__init__()

        # without considering batch size: Input shape : (None,368, 284, 1) , parameters: (3*3*1*16+16) = 160
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=16,  # one input channel gray scale, 16 filters out
            kernel_size=3,
            stride=1,
            padding=1,
        )  # Out:(None,386, 284, 16). ### TRY kernel 7x7 padding 3
        self.conv2 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1
        )  # params: (3*3*16*32+32) = 4640
        self.pool1 = nn.MaxPool2d(2, 2)  # Out: (None, 184, 142, 32)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1
        )  # params: (3*3*16*32+32) = 4640
        self.conv4 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1
        )  # params: (3*3*32*32+32) = 9248
        self.pool2 = nn.MaxPool2d(2, 2)  # Output shape = (None, 92, 71, 64)
        self.bn2 = nn.BatchNorm2d(64)

        # self.conv5 = nn.Conv2d(in_channels = 64, out_channels = 128,
        # kernel_size = 3, stride = 1, padding = 1) # params: (3*3*32*32+32) = 9248
        self.conv6 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1
        )  # params: (3*3*32*32+32) = 9248
        self.pool3 = nn.MaxPool2d(2, 2)  # Output shape = (None, 46, 35, 128)
        self.bn3 = nn.BatchNorm2d(128)
        self.do2 = nn.Dropout(0.3)

        # Fully connected layer
        self.fc1 = nn.Linear(
            128 * 46 * 35, 60
        )  # most recent original size of: 512, 662 -->64 x 82
        self.do3 = nn.Dropout(0.4)  # 40 % probability
        # self.fc3 = nn.Linear(60, 30)
        self.fc2 = nn.Linear(60, 3)  # left with 3 for the three classes

    def forward(self, x):
        """Feed through network."""
        x = self.bn1(self.pool1(F.relu(self.conv2(F.relu(self.conv1(x))))))
        x = self.bn2(self.pool2(F.relu(self.conv4(F.relu(self.conv3(x))))))
        # x = self.bn3(self.pool3(F.relu(self.conv6(F.relu(self.conv5(x))))))
        x = self.bn3(self.pool3(F.relu(self.conv6((x)))))
        x = self.do2(x)
        x = x.view(x.size(0), 128 * 46 * 35)
        x = F.relu(self.fc1(x))
        x = self.do3(x)
        x = self.fc2(x)
        return x


class ConvNetScores(nn.Module):
    """From scratch CNN to score the clocks."""

    def __init__(self):
        """Define CNN."""
        super(ConvNetScores, self).__init__()  # Fixed: was ConvNet, should be ConvNetScores

        # without considering batch size: Input shape : (None,368, 284, 1) , parameters: (3*3*1*16+16) = 160
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=16,  # one input channel gray scale, 16 filters out
            kernel_size=3,
            stride=1,
            padding=1,
        )  # Out:(None,386, 284, 16). ### TRY kernel 7x7 padding 3
        self.conv2 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1
        )  # params: (3*3*16*32+32) = 4640
        self.pool1 = nn.MaxPool2d(2, 2)  # Out: (None, 184, 142, 32)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1
        )  # params: (3*3*16*32+32) = 4640
        self.conv4 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1
        )  # params: (3*3*32*32+32) = 9248
        self.pool2 = nn.MaxPool2d(2, 2)  # Output shape = (None, 92, 71, 64)
        self.bn2 = nn.BatchNorm2d(64)

        # self.conv5 = nn.Conv2d(in_channels = 64, out_channels = 128,
        # kernel_size = 3, stride = 1, padding = 1) # params: (3*3*32*32+32) = 9248
        self.conv6 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1
        )  # params: (3*3*32*32+32) = 9248
        self.pool3 = nn.MaxPool2d(2, 2)  # Output shape = (None, 46, 35, 128)
        self.bn3 = nn.BatchNorm2d(128)
        self.do2 = nn.Dropout(0.3)

        # Fully connected layer
        self.fc1 = nn.Linear(
            128 * 46 * 35, 60
        )  # most recent original size of: 512, 662 -->64 x 82
        self.do3 = nn.Dropout(0.4)  # 40 % probability
        # self.fc3 = nn.Linear(60, 30)
        self.fc2 = nn.Linear(60, 6)  # left with 6 for the score classes

    def forward(self, x):
        """Feed through network."""
        x = self.bn1(self.pool1(F.relu(self.conv2(F.relu(self.conv1(x))))))
        x = self.bn2(self.pool2(F.relu(self.conv4(F.relu(self.conv3(x))))))
        # x = self.bn3(self.pool3(F.relu(self.conv6(F.relu(self.conv5(x))))))
        x = self.bn3(self.pool3(F.relu(self.conv6((x)))))
        x = self.do2(x)
        x = x.view(x.size(0), 128 * 46 * 35)
        x = F.relu(self.fc1(x))
        x = self.do3(x)
        x = self.fc2(x)
        return x
    


# # Create dataset with default Google Drive path
# dataset = ResizedClocks(round=11, round_labels=your_labels, normalize_=True)

# # Or specify a custom path
# dataset = ResizedClocks(
#     round=11, 
#     round_labels=your_labels, 
#     data_path='/your/custom/path/to/clock/drawings',
#     normalize_=True
# )

