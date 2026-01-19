import os
from glob import glob
import pandas as pd
import torch
import torchvision.transforms as transforms
from einops.layers.torch import Rearrange
from PIL import Image
from torch.utils.data import Dataset

FEATURE_FOLDER_ROOT = "./"
SLIDE_FOLDER_ROOT = "./Slide_Path"


class MultiDataSet(Dataset):
    def __init__(self, data, img_batch=800, tasks=None, encoder="unicas", task_id=None):
        if isinstance(data, str) and os.path.isfile(data):
            data = pd.read_csv(data)
        if tasks is None:
            tasks = ["cancer", "candidiasis", "cluecell"]
        self.tasks = list(tasks) if not isinstance(tasks, str) else [tasks]
        if task_id is not None:
            self.data = data[data["task_id"].isin(task_id)]
        else:
            self.data = data 
        self.data = self.data.drop_duplicates()
        self.img_batch = img_batch
        self.encoder = encoder
        feature_folders = glob(f"{FEATURE_FOLDER_ROOT}/Pathology_{encoder}_p*")
        if not feature_folders:
            raise ValueError(f"No feature folder found: {FEATURE_FOLDER_ROOT}/Pathology_{encoder}_p*")
        self.file_folder = feature_folders[0]
        self.columns = self.data.columns.to_list()
        for task in self.tasks:
            assert task in self.columns, f"task names wrong : get {task} -------- "
        self.num_per_cls_list = [
            [len(self.data[self.data[task] == 0]), len(self.data[self.data[task] == 1])]
            for task in self.tasks
        ]
        print(self.encoder, self.tasks, self.num_per_cls_list)
        print(f'total_data:{len(self.data)}----------------------------------------------')

    def __getitem__(self, index):
        patch_tensors = self.get_imgtensor(index)
        label_dict = {}
        
        for i in self.tasks:
            column_index = self.data.columns.get_loc(i)
            label = self.data.iloc[index, column_index]
            label_dict[f"{i}_label"] = label
            
        label_dict["name"] = self.data.iloc[index, 0]
        if "code" in self.data.columns:
            idx = self.data.columns.get_loc("code")
            label_dict["code"] = self.data.iloc[index, idx]
        else:
            label_dict["code"] = str(self.data.iloc[index, 0]).split("/")[-1]
        if "cancer_grade" in self.data.columns:
            grade_index = self.data.columns.get_loc("cancer_grade")
        else:
            grade_index = self.data.columns.get_loc("cancer")
        label_dict["cancer_grade"] = self.data.iloc[index, grade_index]
        if "task_id" in self.data.columns:
            idx = self.data.columns.get_loc("task_id")
            label_dict["task_id"] = self.data.iloc[index, idx]
        else:
            label_dict["task_id"] = -1
        label_dict["num_list"] = self.num_per_cls_list
        labels = label_dict

        return patch_tensors, labels

    def __len__(self):
        return len(self.data)

    @staticmethod
    def collate_fn(batch):
        images, labels = tuple(zip(*batch))
        images = torch.stack(images, dim=0)
        return images, list(labels)

    def get_imgtensor(self, index):
        base_folder = self.file_folder
        folder_path = get_folder_path(self.data.iloc[index, 0], base_folder)
        if folder_path is None:
            raise ValueError(f"Can not find {self.data.iloc[index, 0]}")

        tensor_path = os.path.join(folder_path, "images.pt")
        if not os.path.isfile(tensor_path):
            raise ValueError(f"Tensor of {self.data.iloc[index, 0]} not exist")
        if os.path.getsize(tensor_path) == 0:
            print(folder_path)

        patch_tensors = torch.load(tensor_path, map_location="cpu")
        patch_tensors = patch_tensors[: self.img_batch]
        return patch_tensors


class EncoderData(Dataset):
    def __init__(self, data, img_batch=50, encoder="unicas"):
        if isinstance(data, str) and os.path.isfile(data):
            data = pd.read_csv(data)
        self.data = data.drop_duplicates()
        self.data = self.data.iloc[::-1]
        self.img_batch = img_batch
        self.encoder = encoder
        self.columns = self.data.columns.to_list()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            Rearrange('c (h p1) (w p2) -> (h w) c p1 p2 ', p1=256, p2=256),
            ])

        print(self.encoder)
        print(f'total_data:{len(self.data)}----------------------------------------------')

    def __getitem__(self, index):

        folder_path = get_folder_path(self.data.iloc[index, 0], SLIDE_FOLDER_ROOT)
        if folder_path is None:
            raise ValueError(f"Can not find {self.data.iloc[index, 0]}")

        save_path = folder_path.replace(
            SLIDE_FOLDER_ROOT,
            f"{FEATURE_FOLDER_ROOT}/Pathology_{self.encoder}_patch/",
        )
        if self.transform is not None and os.path.exists(f"{save_path}/images.pt"):
            return torch.zeros(800, 3, 256, 256), save_path


        image_filenames = sorted(
            glob(f"{folder_path}/*.jpg"),
            key=lambda x: os.path.getsize(x),
            reverse=True,
        )
        images = []
        cnt = 0

        for img_name in image_filenames:
            image_path = img_name
            try:
                with Image.open(image_path) as image:
                    image = self.transform(image)
                images.extend(image)
                cnt += 1
            except Exception:
                print(f"{image_path} can not read ----------------")
            if cnt >= self.img_batch:
                break
        
        patch_tensors = torch.stack(images)        
        return patch_tensors, save_path

    def __len__(self):
        return len(self.data)

    @staticmethod
    def collate_fn(batch):
        images, labels = tuple(zip(*batch))
        images = torch.stack(images, dim=0)
        return images, list(labels)


def get_folder_path(name, folder):

    folder_path = None
    if os.path.isdir(os.path.join(folder, name, "torch")):
        folder_path = os.path.join(folder, name, "torch")
    elif os.path.isdir(os.path.join(folder, "yangxing", "Torch", name, "torch")):
        folder_path = os.path.join(folder, "yangxing", "Torch", name, "torch")
    elif os.path.isdir(os.path.join(folder, "yinxing", "Torch", name, "torch")):
        folder_path = os.path.join(folder, "yinxing", "Torch", name, "torch")
    elif os.path.isdir(os.path.join(folder, name)):
        folder_path = os.path.join(folder, name)
    elif os.path.isdir(os.path.join(folder, "Torch", name, "torch")):
        folder_path = os.path.join(folder, "Torch", name, "torch")
    elif os.path.isdir(os.path.join(folder, "Data2025", name, "torch")):
        folder_path = os.path.join(folder, "Data2025", name, "torch")

    if folder_path is None:
        print(f"{name} not in {folder}")
    return folder_path



