import pandas as pd
import numpy as np
import io
import torch
from PIL import Image
import os

def load_data():
    """Load stata data from Google Drive, rounds 1 - 10; contains diagnoses, scoring, subject ID to map to images
    hc1disescn9 : 1 - YES to dementia/Alzheimers, 2 - NO Dementia, may want to drop -9 and -1?, may need to relabel 7.
    cg1dclkdraw: score of drawing
    spid: Subject ID, number maps to number in image file names"""

    values = [11]  # Adjust to [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11] if needed
    round_data = pd.DataFrame()

    for val in values:
        # Construct file path in Google Drive
        file_path = f'/content/drive/MyDrive/Nhats Dataset/NHATS_R11_Final_Release_STATA_V2/NHATS_Round_{val}_SP_File_V2.dta'
        
        # Check if file exists
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue

        # Read stata file from Google Drive
        try:
            data = pd.io.stata.read_stata(file_path)
            columns = ["spid", f"cg{val}dclkdraw", f"hc{val}disescn9"]
            # Select only available columns
            data = data[[col for col in columns if col in data.columns]]
            data["round"] = val

            # Rename columns
            data.rename(
                columns={
                    f"cg{val}dclkdraw": "cg1dclkdraw",
                    f"hc{val}disescn9": "hc1disescn9",
                },
                inplace=True,
            )
            round_data = pd.concat([round_data, data], ignore_index=True)
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")
            continue

    return round_data

def load_images():
    """Currently loading one image at a time and turning the bool matrix into the inverse
    numpy array"""
    # Construct file path in Google Drive
    file_path = '/content/gdrive/MyDrive/Data/Nhats Dataset/NHATS_R11_ClockDrawings_V2/10000008.tif'

    # Check if file exists
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None

    # Open image directly from Google Drive
    try:
        im = Image.open(file_path)
        imarray = np.logical_not(np.array(im)).astype(int)
        return imarray
    except Exception as e:
        print(f"Error loading image {file_path}: {e}")
        return None

def hats_load_data():
    """This loads in the data with all the columns needed for label creation using the
    NHATs Dementia Classification criteria. It loads diagnosis variables, ID, round number,
    and Cognitive test variables for the domains of Orientation, Memory and Executive Functioning."""

    values = [11]  # Adjust to [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11] if needed
    round_data = pd.DataFrame()

    for val in values:
        # Construct file path in Google Drive
        file_path = f'/content/drive/MyDrive/Nhats Dataset/NHATS_R11_Final_Release_STATA_V2/NHATS_Round_{val}_SP_File_V2.dta'
        
        # Check if file exists
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue

        # Read stata file from Google Drive
        try:
            data = pd.io.stata.read_stata(file_path)

            if val == 1:
                columns = [
                    "spid",
                    f"cg{val}dclkdraw",
                    f"hc{val}disescn9",
                    f"cg{val}presidna1",
                    f"cg{val}presidna3",
                    f"cg{val}vpname1",
                    f"cg{val}vpname3",
                    f"cg{val}todaydat1",
                    f"cg{val}todaydat2",
                    f"cg{val}todaydat3",
                    f"cg{val}todaydat4",
                    f"cg{val}dwrdimmrc",
                    f"cg{val}dwrddlyrc",
                ]
            else:
                columns = [
                    "spid",
                    f"cg{val}dclkdraw",
                    f"hc{val}disescn9",
                    f"cp{val}dad8dem",
                    f"cg{val}presidna1",
                    f"cg{val}presidna3",
                    f"cg{val}vpname1",
                    f"cg{val}vpname3",
                    f"cg{val}todaydat1",
                    f"cg{val}todaydat2",
                    f"cg{val}todaydat3",
                    f"cg{val}todaydat4",
                    f"cg{val}dwrdimmrc",
                    f"cg{val}dwrddlyrc",
                ]

            # Select only available columns
            data = data[[col for col in columns if col in data.columns]]
            data["round"] = val

            # Rename columns
            data.rename(
                columns={
                    f"cg{val}dclkdraw": "cg1dclkdraw",
                    f"hc{val}disescn9": "hc1disescn9",
                    f"cp{val}dad8dem": "cp1dad8dem",
                    f"cg{val}presidna1": "cg1presidna1",
                    f"cg{val}presidna3": "cg1presidna3",
                    f"cg{val}vpname1": "cg1vpname1",
                    f"cg{val}vpname3": "cg1vpname3",
                    f"cg{val}todaydat1": "cg1todaydat1",
                    f"cg{val}todaydat2": "cg1todaydat2",
                    f"cg{val}todaydat3": "cg1todaydat3",
                    f"cg{val}todaydat4": "cg1todaydat4",
                    f"cg{val}dwrdimmrc": "cg1dwrdimmrc",
                    f"cg{val}dwrddlyrc": "cg1dwrddlyrc",
                },
                inplace=True,
            )
            round_data = pd.concat([round_data, data], ignore_index=True)
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")
            continue

    return round_data

def load_np_files(data, target):
    """Takes in filenames for image data and target data
    expands the dimensions for the image data to reflect grayscale
    of dimension 1, prepping for the dataloader, and then zips the
    image data and labels together for future dataloading. Also returns
    label tensors for each split"""

    try:
        # Get data
        x_data = np.load(data)
        y_data = np.load(target)

        # Need to add that extra dimension for grayscale depth of 1 channel
        x_data = np.expand_dims(x_data, 1)
        print(x_data.shape)

        # Zip image data and labels together
        data = [(x, y) for x, y in zip(x_data, y_data)]

        # Turn targets into tensors
        y_tensor = torch.from_numpy(y_data)

        return data, y_tensor
    except Exception as e:
        print(f"Error in load_np_files: {e}")
        return [], None
