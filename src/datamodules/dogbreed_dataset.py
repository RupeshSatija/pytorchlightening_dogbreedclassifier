import os
import shutil
import zipfile
from typing import Optional

import gdown
import lightning.pytorch as pl
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder

IMAGE_SIZE = 150
CROP_SIZE = 100


class DogBreedDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "dataset",
        batch_size: int = 32,
        num_workers: int = 4,
        val_split: float = 0.2,
        test_split: float = 0.1,
        google_drive_id: str = "1X4a5jGErxXJZ0mdNBZHhpytacEj-wCRU",
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.test_split = test_split
        self.google_drive_id = google_drive_id
        self.train_dataset = self.val_dataset = self.test_dataset = None
        self.class_names = None

    def prepare_data(self):
        # Download and extract the dataset if it doesn't exist
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir, exist_ok=True)
            zip_path = os.path.join(self.data_dir, "dog_breeds.zip")

            # Download the zip file
            gdown.download(id=self.google_drive_id, output=zip_path, quiet=False)

            # Extract the zip file
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(self.data_dir)

            # Remove the zip file
            os.remove(zip_path)

            # Move contents one level up if needed
            extracted_dir = os.path.join(self.data_dir, "dataset")
            if os.path.exists(extracted_dir):
                for item in os.listdir(extracted_dir):
                    s = os.path.join(extracted_dir, item)
                    d = os.path.join(self.data_dir, item)
                    shutil.move(s, d)
                os.rmdir(extracted_dir)

    @property
    def normalize_transform(self):
        return transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    @property
    def train_transform(self):
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(CROP_SIZE),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                ),
                transforms.RandomRotation(20),
                transforms.ToTensor(),
                self.normalize_transform,
            ]
        )

    @property
    def valid_transform(self):
        return transforms.Compose(
            [
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.CenterCrop(CROP_SIZE),
                transforms.ToTensor(),
                self.normalize_transform,
            ]
        )

    def setup(self, stage: Optional[str] = None):
        # Ensure data is prepared
        self.prepare_data()

        # Create the full dataset
        full_dataset = ImageFolder(self.data_dir, transform=self.train_transform)

        # Store the class names
        self.class_names = full_dataset.classes

        # Calculate split sizes
        total_size = len(full_dataset)
        val_size = int(self.val_split * total_size)
        test_size = int(self.test_split * total_size)
        train_size = total_size - val_size - test_size

        # Split the dataset
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            full_dataset, [train_size, val_size, test_size]
        )

        # Apply transforms to validation and test datasets
        self.val_dataset.dataset.transform = self.valid_transform
        self.test_dataset.dataset.transform = self.valid_transform

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def get_class_names(self):
        return self.class_names


# # import os
# # import zipfile
# # from typing import Optional

# # import gdown
# # import pytorch_lightning as pl
# # from PIL import Image
# # from torch.utils.data import DataLoader, Dataset
# # from torchvision import transforms

# # IMAGE_SIZE = 150
# # CROP_SIZE = 100


# # class DogBreedDataset(Dataset):
# #     def __init__(self, data_dir: str, transform: Optional[transforms.Compose] = None):
# #         self.data_dir = data_dir
# #         self.transform = transform or transforms.Compose(
# #             [
# #                 transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
# #                 transforms.ToTensor(),
# #                 transforms.Normalize(
# #                     mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
# #                 ),
# #             ]
# #         )
# #         self.classes = sorted(os.listdir(data_dir))
# #         self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
# #         self.images = self._load_images()

# #     def _load_images(self):
# #         images = []
# #         for cls_name in self.classes:
# #             cls_dir = os.path.join(self.data_dir, cls_name)
# #             for img_name in os.listdir(cls_dir):
# #                 img_path = os.path.join(cls_dir, img_name)
# #                 images.append((img_path, self.class_to_idx[cls_name]))
# #         return images

# #     def __len__(self):
# #         return len(self.images)

# #     def __getitem__(self, idx):
# #         img_path, label = self.images[idx]
# #         image = Image.open(img_path).convert("RGB")

# #         if self.transform:
# #             image = self.transform(image)

# #         return image, label


# # class DogBreedDataModule(pl.LightningDataModule):
# #     def __init__(
# #         self,
# #         data_dir: str = "data",
# #         batch_size: int = 32,
# #         num_workers: int = 4,
# #         google_drive_id: str = "1X4a5jGErxXJZ0mdNBZHhpytacEj-wCRU",
# #     ):
# #         super().__init__()
# #         self.data_dir = data_dir
# #         self.batch_size = batch_size
# #         self.num_workers = num_workers
# #         self.google_drive_id = google_drive_id
# #         self.train_dataset = None
# #         self.val_dataset = None
# #         self.test_dataset = None
# #         self.transform = transforms.Compose(
# #             [
# #                 transforms.RandomResizedCrop(CROP_SIZE),
# #                 transforms.RandomHorizontalFlip(),
# #                 transforms.ToTensor(),
# #                 transforms.Normalize(
# #                     mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
# #                 ),
# #             ]
# #         )

# #     def prepare_data(self):
# #         if not os.path.exists(self.data_dir):
# #             os.makedirs(self.data_dir, exist_ok=True)
# #             zip_path = os.path.join(self.data_dir, "dog-breed-image-dataset.zip")

# #             if not os.path.exists(zip_path):
# #                 try:
# #                     gdown.download(
# #                         id=self.google_drive_id, output=zip_path, quiet=False
# #                     )
# #                 except gdown.exceptions.FileURLRetrievalError as e:
# #                     print(f"Error downloading file: {e}")
# #                     print("Attempting to download using the full URL...")
# #                     full_url = f"https://drive.google.com/uc?id={file_id}"
# #                     gdown.download(url=full_url, output=zip_path, quiet=False)

# #             with zipfile.ZipFile(zip_path, "r") as zip_ref:
# #                 zip_ref.extractall(self.data_dir)

# #             # gdown.extractall(zip_path, self.data_dir)
# #             os.remove(zip_path)

# #     def setup(self, stage: Optional[str] = None):
# #         if stage == "fit" or stage is None:
# #             self.train_dataset = DogBreedDataset(
# #                 os.path.join(self.data_dir, "train"), transform=self.transform
# #             )
# #             self.val_dataset = DogBreedDataset(
# #                 os.path.join(self.data_dir, "val"),
# #                 transform=transforms.Compose(
# #                     [
# #                         transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
# #                         transforms.ToTensor(),
# #                         transforms.Normalize(
# #                             mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
# #                         ),
# #                     ]
# #                 ),
# #             )

# #         if stage == "test" or stage is None:
# #             self.test_dataset = DogBreedDataset(
# #                 os.path.join(self.data_dir, "test"),
# #                 transform=transforms.Compose(
# #                     [
# #                         transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
# #                         transforms.ToTensor(),
# #                         transforms.Normalize(
# #                             mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
# #                         ),
# #                     ]
# #                 ),
# #             )

# #     def train_dataloader(self):
# #         if self.train_dataset is None:
# #             self.setup(stage="fit")
# #         return DataLoader(
# #             self.train_dataset,
# #             batch_size=self.batch_size,
# #             shuffle=True,
# #             num_workers=self.num_workers,
# #             pin_memory=True,
# #         )

# #     def val_dataloader(self):
# #         if self.val_dataset is None:
# #             self.setup(stage="fit")
# #         return DataLoader(
# #             self.val_dataset,
# #             batch_size=self.batch_size,
# #             shuffle=False,
# #             num_workers=self.num_workers,
# #             pin_memory=True,
# #         )

# #     def test_dataloader(self):
# #         if self.test_dataset is None:
# #             self.setup(stage="test")
# #         return DataLoader(
# #             self.test_dataset,
# #             batch_size=self.batch_size,
# #             shuffle=False,
# #             num_workers=self.num_workers,
# #             pin_memory=True,
# #         )

# import os
# import shutil
# import zipfile
# from typing import Optional

# import gdown
# import lightning.pytorch as pl
# from torch.utils.data import DataLoader, random_split
# from torchvision import transforms
# from torchvision.datasets import ImageFolder

# IMAGE_SIZE = 150
# CROP_SIZE = 100


# class DogBreedDataModule(pl.LightningDataModule):
#     def __init__(
#         self,
#         data_dir: str = "data",
#         batch_size: int = 32,
#         num_workers: int = 4,
#         val_split: float = 0.2,
#         test_split: float = 0.1,
#     ):
#         super().__init__()
#         self.data_dir = data_dir
#         self.batch_size = batch_size
#         self.num_workers = num_workers
#         self.val_split = val_split
#         self.test_split = test_split
#         self.google_drive_id = "1X4a5jGErxXJZ0mdNBZHhpytacEj-wCRU"
#         self._data_prepared = False
#         self.train_dataset = self.val_dataset = self.test_dataset = None

#     def _apply_transform(self, dataset, transform):
#         return [(transform(img), label) for img, label in dataset]

#     def prepare_data(self):
#         # Download and extract the dataset if it doesn't exist
#         if not os.path.exists(self.data_dir):
#             os.makedirs(self.data_dir, exist_ok=True)
#             zip_path = os.path.join(self.data_dir, "dog_breeds.zip")

#             # Download the zip file
#             gdown.download(id=self.google_drive_id, output=zip_path, quiet=False)

#             # Extract the zip file
#             with zipfile.ZipFile(zip_path, "r") as zip_ref:
#                 zip_ref.extractall(self.data_dir)

#             # Remove the zip file
#             os.remove(zip_path)

#         # Assume extracted data is in a subfolder named 'dataset'
#         extracted_dir = os.path.join(self.data_dir, "dataset")
#         if os.path.exists(extracted_dir):
#             # Move contents up one level
#             for item in os.listdir(extracted_dir):
#                 shutil.move(os.path.join(extracted_dir, item), self.data_dir)
#             os.rmdir(extracted_dir)

#     @property
#     def normalize_transform(self):
#         return transforms.Normalize(
#             mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
#         )

#     @property
#     def train_transform(self):
#         return transforms.Compose(
#             [
#                 transforms.RandomResizedCrop(CROP_SIZE),
#                 transforms.RandomHorizontalFlip(),
#                 transforms.ColorJitter(
#                     brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
#                 ),
#                 transforms.RandomRotation(20),
#                 transforms.ToTensor(),
#                 self.normalize_transform,
#             ]
#         )

#     @property
#     def valid_transform(self):
#         return transforms.Compose(
#             [
#                 transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
#                 transforms.CenterCrop(CROP_SIZE),
#                 transforms.ToTensor(),
#                 self.normalize_transform,
#             ]
#         )

#     def setup(self, stage: Optional[str] = None):
#         if not self._data_prepared:
#             self.prepare_data()
#             self._data_prepared = True

#         # Create the full dataset
#         full_dataset = ImageFolder(self.data_dir)

#         # Calculate split sizes
#         total_size = len(full_dataset)
#         val_size = int(self.val_split * total_size)
#         test_size = int(self.test_split * total_size)
#         train_size = total_size - val_size - test_size
#         # Split the dataset

#         self.train_dataset, self.val_dataset, self.test_dataset = random_split(
#             full_dataset, [train_size, val_size, test_size]
#         )

#         # Apply transforms
#         self.train_dataset = self._apply_transform(
#             self.train_dataset, self.train_transform
#         )
#         self.val_dataset = self._apply_transform(self.val_dataset, self.valid_transform)
#         self.test_dataset = self._apply_transform(
#             self.test_dataset, self.valid_transform
#         )

#         # Store the class names
#         self.class_names = self.train_dataset.classes

#     def train_dataloader(self):
#         if self.train_dataset is None:
#             self.setup(stage="fit")
#         return DataLoader(
#             self.train_dataset,
#             batch_size=self.batch_size,
#             shuffle=True,
#             num_workers=self.num_workers,
#         )

#     def val_dataloader(self):
#         return DataLoader(
#             self.val_dataset,
#             batch_size=self.batch_size,
#             shuffle=False,
#             num_workers=self.num_workers,
#         )

#     def test_dataloader(self):
#         return DataLoader(
#             self.test_dataset,
#             batch_size=self.batch_size,
#             shuffle=False,
#             num_workers=self.num_workers,
#         )

#     def get_class_names(self):
#         return self.class_names
