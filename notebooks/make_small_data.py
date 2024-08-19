# import os
# os.chdir('../6DoF')

import sys
sys.path.append("../6DoF")

import torchvision
from torchvision import transforms
from dataset import ObjaverseData
import torch
from tqdm import tqdm
import os

image_transforms = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ]
)

root_dir = "/data2/wlsgur4011/zero123_data/views_release/"
new_dir = root_dir.replace("zero123_data", "zero123_data_small")

# 1. copy npz files
# os.system(f"rsync -avm --include='*.npy' --include='*/' --exclude='*' {root_dir}/ {new_dir}/")

# 2. prepocess images
for is_validation in [True, False]:
    train_dataset = ObjaverseData(root_dir=root_dir,
                                  image_transforms=image_transforms,
                                  validation=is_validation,
                                  T_in=0,
                                  T_out=1,
                                  save_preprocessed=True,
                                  use_preprocessed=False,
                                  )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=False,
        batch_size=16,
        num_workers=16,
    )

    for data in tqdm(train_dataloader):
        pass

    # 3. check new_dataset working
    new_dataset = ObjaverseData(root_dir=new_dir,
                                image_transforms=image_transforms,
                                validation=is_validation,
                                T_in=1,
                                T_out=1,
                                fix_sample=True,
                                use_preprocessed=True,
                                )
    old_dataset = ObjaverseData(root_dir=root_dir,
                                image_transforms=image_transforms,
                                validation=is_validation,
                                T_in=1,
                                T_out=1,
                                fix_sample=True,
                                use_preprocessed=False,
                                )

    for idx in range(100):
        new_data = new_dataset[idx]["image_input"]
        old_data = old_dataset[idx]["image_input"]
        if not torch.all(new_data == old_data):
            raise ValueError("new_dataset not working")
    print("new_dataset working")
