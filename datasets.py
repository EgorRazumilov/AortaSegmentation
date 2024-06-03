import torch
from torch.utils.data import Dataset
import nibabel as nib
import os
import numpy as np
import nrrd
from pydicom import dcmread
from scipy.ndimage import zoom

import albumentations as A

from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    ToTensord,
    Spacingd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    NormalizeIntensityd,
    RandSpatialCropd
)

class BTCV2DDataset(Dataset):
    def __init__(self,
                 images_path,
                 labels_path,
                 start,
                 stop):
        self.images_path = images_path
        self.labels_path = labels_path

        self.images_depths = []
        self.images_list = sorted(os.listdir(images_path))[start:stop]
        for file in self.images_list:
            full_path = os.path.join(images_path, file)
            img = nib.load(full_path)
            self.images_depths.append(img.shape[-1])
    
    def __len__(self):
        return sum(self.images_depths)
    
    def __getitem__(self, index):
        cur_image_id = 0
        cur_index = index
        while cur_index >= self.images_depths[cur_image_id]:
            cur_index -= self.images_depths[cur_image_id]
            cur_image_id += 1
        img_path = os.path.join(self.images_path, self.images_list[cur_image_id])
        img = nib.load(img_path)
        slice = torch.tensor(img.get_fdata()[:, :, cur_index], dtype=torch.float32)

        slice = (slice - slice.min()) / (slice.max() - slice.min())

        label_path = self.images_list[cur_image_id].replace('img', 'label')
        label_path = os.path.join(self.labels_path, label_path)
        label = nib.load(label_path)
        slice_label = torch.tensor(label.get_fdata()[:, :, cur_index] == 8, dtype=torch.float32)
        return slice[None], slice_label[None]

    
class AVT2DDataset(Dataset):
    def __init__(self, path_to_data):
        clinics = ['KiTS', 'Rider', 'Dongyang']
        self.images_depths = []
        self.patients = []
        for clinic in clinics:
            path_to_clinic = os.path.join(path_to_data, clinic)
            for patient in os.listdir(path_to_clinic):
                patient_name = os.listdir(os.path.join(path_to_clinic, patient))[0].split('.')[0]
                sample_path = os.path.join(path_to_clinic, patient, patient_name + '.nrrd')
                sample, _ = nrrd.read(sample_path)
                self.images_depths.append(sample.shape[-1])
                self.patients.append(os.path.join(path_to_clinic, patient, patient_name))
    
    def __len__(self):
        return sum(self.images_depths)
    
    def __getitem__(self, index):
        cur_image_id = 0
        cur_index = index
        while cur_index >= self.images_depths[cur_image_id]:
            cur_index -= self.images_depths[cur_image_id]
            cur_image_id += 1
        img_path = self.patients[cur_image_id] + '.nrrd'
        img, _ = nrrd.read(img_path)
        slice = torch.tensor(img[:, :, cur_index].astype(float), dtype=torch.float32)
        
        slice = (slice - slice.min()) / (slice.max() - slice.min())

        label_path = self.patients[cur_image_id] + '.seg.nrrd'
        label, _ = nrrd.read(label_path)
        slice_label = torch.tensor(label[:, :, cur_index], dtype=torch.float32)
        return slice[None], slice_label[None]



INPUT_TRANSFORMS = Compose(
    [
        LoadImaged(keys=['image', 'label']),
        EnsureChannelFirstd(keys=['image', 'label']),
        Spacingd(keys=['image', 'label'], pixdim=(1., 1., 1.), mode=("bilinear", "nearest")),
        NormalizeIntensityd(keys=['image'])
    ]
)

TRAIN_TRANSFORMS = Compose(
    [
        RandSpatialCropd(keys=['image', 'label'], roi_size=(128, 128, 128)),
        RandShiftIntensityd(keys=['image'], offsets=0.1),
        RandScaleIntensityd(keys=['image'], factors=[0.9, 1.1]),
        ToTensord(keys=['image', 'label'])
    ]
)

VAL_TRANSFORMS = Compose(
    [
        ToTensord(keys=['image', 'label'])
    ]
)


class AVT3DDataset(Dataset):
    def __init__(self, path_to_data, is_train, start, stop, seed=42):
        clinics = ['KiTS', 'Rider', 'Dongyang']
        patients = []
        for clinic in clinics:
            path_to_clinic = os.path.join(path_to_data, clinic)
            for patient in sorted(os.listdir(path_to_clinic)):
                path_to_patient = os.path.join(path_to_clinic, patient)
                patients.append(path_to_patient)
        ids = np.random.RandomState(seed=seed).permutation(len(patients))
        files_tuples = []
        for patient_path in patients:
            files_tuple = ['', '']
            for file in os.listdir(patient_path):
                if '.seg' in file:
                    files_tuple[1] = os.path.join(patient_path, file)
                else:
                    files_tuple[0] = os.path.join(patient_path, file)
            files_tuples.append(tuple(files_tuple))
        self.img_mask_paths = [files_tuples[x] for x in ids[start:stop]]
        
        self.is_train = is_train
        self.inputs = []
        for i in range(len(self.img_mask_paths)):
            image_path = self.img_mask_paths[i][0]
            label_path = self.img_mask_paths[i][1]
            self.inputs.append(INPUT_TRANSFORMS({'image': image_path, 'label': label_path}))

        if is_train:
            self.transforms = TRAIN_TRANSFORMS
        else:
            self.transforms = VAL_TRANSFORMS
    
    def __len__(self):
        if not self.is_train:
            return len(self.inputs)
        return 7000
    
    def __getitem__(self, index):        
        input = self.inputs[index % len(self.inputs)]
        out = self.transforms(input)
        return out['image'], (out['label']).float()

class BTCV3DDataset(Dataset):
    def __init__(self, images_path, labels_path, is_train, start, stop):
        self.images_list = []
        self.labels_list = []
        for x in sorted(os.listdir(images_path))[start:stop]:
            self.images_list.append(os.path.join(images_path, x))
            self.labels_list.append(os.path.join(labels_path, x.replace('img', 'label')))
            
        self.is_train = is_train
        self.inputs = []
        for i in range(len(self.images_list)):
            image_path = self.images_list[i]
            label_path = self.labels_list[i]
            self.inputs.append(INPUT_TRANSFORMS({'image': image_path, 'label': label_path}))
            
        if is_train:
            self.transforms = TRAIN_TRANSFORMS
        else:
            self.transforms = VAL_TRANSFORMS
            
    def __len__(self):
        if not self.is_train:
            return len(self.images_list)
        return 5000
    
    def __getitem__(self, index):        
        input = self.inputs[index % len(self.images_list)]
        out = self.transforms(input)
        return out['image'], (out['label'] == 8).float()


class SegThor2DDataset(Dataset):
    def __init__(self,
                 images_path,
                 is_train,
                 start,
                 stop):
        self.images_path = images_path

        self.images_depths = []
        self.images_list = sorted(os.listdir(images_path))[start:stop]
        for file in self.images_list:
            full_path = os.path.join(images_path, file, file + '.nii.gz')
            img = nib.load(full_path)
            self.images_depths.append(img.shape[-1])
        if is_train:
            self.transform = A.Compose([
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.RandomRotate90(p=0.5),
                    A.GaussNoise(p=0.2)
            ])
        else:
            self.transform = None
    def __len__(self):
        return sum(self.images_depths)
    
    def __getitem__(self, index):
        cur_image_id = 0
        cur_index = index
        while cur_index >= self.images_depths[cur_image_id]:
            cur_index -= self.images_depths[cur_image_id]
            cur_image_id += 1
        img_path = os.path.join(self.images_path, self.images_list[cur_image_id], self.images_list[cur_image_id] + '.nii.gz')
        img = nib.load(img_path)
        slice = torch.tensor(img.get_fdata()[:, :, cur_index], dtype=torch.float32)
        
        slice = (slice - slice.min()) / (slice.max() - slice.min())
        
        label_path = os.path.join(self.images_path, self.images_list[cur_image_id], 'GT.nii.gz')
        label = nib.load(label_path)
        slice_label = torch.tensor(label.get_fdata()[:, :, cur_index] == 4, dtype=torch.float32)

        if self.transform is not None:
            transformed = self.transform(image=slice.numpy(), mask=slice_label.numpy())
            slice = torch.from_numpy(transformed['image'])
            slice_label = torch.from_numpy(transformed['mask'])
        return slice[None], slice_label[None]

class SegThor3DDataset(Dataset):
    def __init__(self, images_path, is_train, start, stop):
        self.images_list = []
        self.labels_list = []
        for file in sorted(os.listdir(images_path))[start:stop]:
            full_image_path = os.path.join(images_path, file, file + '.nii.gz')
            full_label_path = os.path.join(images_path, file, 'GT.nii.gz')
            self.images_list.append(full_image_path)
            self.labels_list.append(full_label_path)
            
        self.is_train = is_train
        self.inputs = []
        for i in range(len(self.images_list)):
            image_path = self.images_list[i]
            label_path = self.labels_list[i]
            self.inputs.append(INPUT_TRANSFORMS({'image': image_path, 'label': label_path}))
            
        if is_train:
            self.transforms = TRAIN_TRANSFORMS
        else:
            self.transforms = VAL_TRANSFORMS
            
    def __len__(self):
        if not self.is_train:
            return len(self.images_list)
        return 5000
    
    def __getitem__(self, index):        
        input = self.inputs[index % len(self.images_list)]
        out = self.transforms(input)
        return out['image'], (out['label'] == 4).float()

class SegThor3DDataset(Dataset):
    def __init__(self, images_path, is_train, start, stop):
        self.images_list = []
        self.labels_list = []
        for file in sorted(os.listdir(images_path))[start:stop]:
            full_image_path = os.path.join(images_path, file, file + '.nii.gz')
            full_label_path = os.path.join(images_path, file, 'GT.nii.gz')
            self.images_list.append(full_image_path)
            self.labels_list.append(full_label_path)
            
        self.is_train = is_train
        self.inputs = []
        for i in range(len(self.images_list)):
            image_path = self.images_list[i]
            label_path = self.labels_list[i]
            self.inputs.append(INPUT_TRANSFORMS({'image': image_path, 'label': label_path}))
            
        if is_train:
            self.transforms = TRAIN_TRANSFORMS
        else:
            self.transforms = VAL_TRANSFORMS
            
    def __len__(self):
        if not self.is_train:
            return len(self.images_list)
        return 5000
    
    def __getitem__(self, index):        
        input = self.inputs[index % len(self.images_list)]
        out = self.transforms(input)
        return out['image'], (out['label'] == 4).float()


