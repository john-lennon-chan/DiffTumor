from monai.transforms import (
    AsDiscrete,
    AsChannelFirstd,
    AddChanneld,
    Compose,
    ConcatItemsd,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    ToTensord,
    CenterSpatialCropd,
    Resized,
    SpatialPadd,
    apply_transform,
    RandZoomd,
    RandCropByLabelClassesd,
)
import sys
from copy import copy, deepcopy
import h5py, os
import numpy as np
import torch
from typing import IO, TYPE_CHECKING, Any, Callable, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple, Union


sys.path.append("..") 

from torch.utils.data import Subset

from monai.data import DataLoader, Dataset, list_data_collate, DistributedSampler, CacheDataset
from monai.data.utils import pad_list_data_collate
from monai.config import DtypeLike, KeysCollection
from monai.transforms.transform import Transform, MapTransform
from monai.utils.enums import TransformBackends
from monai.config.type_definitions import NdarrayOrTensor
from monai.transforms.io.array import LoadImage, SaveImage
from monai.utils import GridSamplePadMode, ensure_tuple, ensure_tuple_rep
from monai.data.image_reader import ImageReader
from monai.utils.enums import PostFix
DEFAULT_POST_FIX = PostFix.meta()

class DiskDataset(Dataset):
    def __init__(self, data, transform=None, save_dir=None):
        super().__init__(data, transform)
        self.save_dir = save_dir
        if self.save_dir is not None:
            os.makedirs(self.save_dir, exist_ok=True)

    def __getitem__(self, index):
        #print(f"processing file {index}")
        filename = f"data_{index}.npz"
        filepath = os.path.join(self.save_dir, filename)

        try:
            loaded = np.load(filepath)
            data = [{'image': arr} for arr in loaded.values()]
        except:
            data = super().__getitem__(index)
            if self.save_dir is not None:
                try:
                    np.savez_compressed(filepath, a = data[0]['image'], b = data[1]['image'])
                except Exception as e:
                    print(e)
                    print(f"failed to save {filepath}")

        #print(f"Two data identical? {data[0]['image'] == data[1]['image']}")
        #print(f"data type is {type(data)}")
        #print(f"each element of that list is a {[type(i) for i in data]}")
        #tobeprinted = {k: type(v) for k, v in data[0].items()}
        #print(f"each element of the first element of that list is a {tobeprinted}")


        return data


class UniformDataset(Dataset):
    def __init__(self, data, transform, datasetkey):
        super().__init__(data=data, transform=transform)
        self.dataset_split(data, datasetkey)
        self.datasetkey = datasetkey
    
    def dataset_split(self, data, datasetkey):
        self.data_dic = {}
        for key in datasetkey:
            self.data_dic[key] = []
        for img in data:
            key = get_key(img['name'])
            self.data_dic[key].append(img)
        
        self.datasetnum = []
        for key, item in self.data_dic.items():
            assert len(item) != 0, f'the dataset {key} has no data'
            self.datasetnum.append(len(item))
        self.datasetlen = len(datasetkey)
    
    def _transform(self, set_key, data_index):
        data_i = self.data_dic[set_key][data_index]
        return apply_transform(self.transform, data_i) if self.transform is not None else data_i
    
    def __getitem__(self, index):
        ## the index generated outside is only used to select the dataset
        ## the corresponding data in each dataset is selelcted by the np.random.randint function
        set_index = index % self.datasetlen
        set_key = self.datasetkey[set_index]
        data_index = np.random.randint(self.datasetnum[set_index], size=1)[0]
        return self._transform(set_key, data_index)


class UniformCacheDataset(CacheDataset):
    def __init__(self, data, transform, cache_rate, datasetkey):
        super().__init__(data=data, transform=transform, cache_rate=cache_rate)
        self.datasetkey = datasetkey
        self.data_statis()
    
    def data_statis(self):
        data_num_dic = {}
        for key in self.datasetkey:
            data_num_dic[key] = 0
        for img in self.data:
            key = get_key(img['name'])
            data_num_dic[key] += 1

        self.data_num = []
        for key, item in data_num_dic.items():
            assert item != 0, f'the dataset {key} has no data'
            self.data_num.append(item)
        
        self.datasetlen = len(self.datasetkey)
    
    def index_uniform(self, index):
        ## the index generated outside is only used to select the dataset
        ## the corresponding data in each dataset is selelcted by the np.random.randint function
        set_index = index % self.datasetlen
        data_index = np.random.randint(self.data_num[set_index], size=1)[0]
        post_index = int(sum(self.data_num[:set_index]) + data_index)
        return post_index

    def __getitem__(self, index):
        post_index = self.index_uniform(index)
        return self._transform(post_index)
class LoadImaged_BodyMap(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        reader: Optional[Union[ImageReader, str]] = None,
        dtype: DtypeLike = np.float32,
        meta_keys: Optional[KeysCollection] = None,
        meta_key_postfix: str = DEFAULT_POST_FIX,
        overwriting: bool = False,
        image_only: bool = False,
        ensure_channel_first: bool = False,
        simple_keys: bool = False,
        allow_missing_keys: bool = False,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self._loader = LoadImage(reader, image_only, dtype, ensure_channel_first, simple_keys, *args, **kwargs)
        if not isinstance(meta_key_postfix, str):
            raise TypeError(f"meta_key_postfix must be a str but is {type(meta_key_postfix).__name__}.")
        self.meta_keys = ensure_tuple_rep(None, len(self.keys)) if meta_keys is None else ensure_tuple(meta_keys)
        if len(self.keys) != len(self.meta_keys):
            raise ValueError("meta_keys should have the same length as keys.")
        self.meta_key_postfix = ensure_tuple_rep(meta_key_postfix, len(self.keys))
        self.overwriting = overwriting


    def register(self, reader: ImageReader):
        self._loader.register(reader)


    def __call__(self, data, reader: Optional[ImageReader] = None):
        d = dict(data)
        # print(d['image'])
        for key, meta_key, meta_key_postfix in self.key_iterator(d, self.meta_keys, self.meta_key_postfix):
            try:
                data = self._loader(d[key], reader)
            except:
                print(d['name'])
            if self._loader.image_only:
                d[key] = data
            else:
                if not isinstance(data, (tuple, list)):
                    raise ValueError("loader must return a tuple or list (because image_only=False was used).")
                d[key] = data[0]
                if not isinstance(data[1], dict):
                    raise ValueError("metadata must be a dict.")
                meta_key = meta_key or f"{key}_{meta_key_postfix}"
                if meta_key in d and not self.overwriting:
                    raise KeyError(f"Metadata with key {meta_key} already exists and overwriting=False.")
                d[meta_key] = data[1]
        d['label'], d['label_meta_dict'] = self.label_transfer(d['label'], d['ct'].shape)
        #print(f"shape is {d['ct'].shape}, label shape is {d['label'].shape}")
        return d

    def label_transfer(self, lbl_dir, shape):
        organ_lbl = np.zeros(shape)

        """
        if os.path.exists(lbl_dir + 'liver' + '.nii.gz'):
            array, mata_infomation = self._loader(lbl_dir + 'liver' + '.nii.gz')
            organ_lbl[array > 0] = 1
        if os.path.exists(lbl_dir + 'pancreas' + '.nii.gz'):
            array, mata_infomation = self._loader(lbl_dir + 'pancreas' + '.nii.gz')
            organ_lbl[array > 0] = 2
        if os.path.exists(lbl_dir + 'kidney_left' + '.nii.gz'):
            array, mata_infomation = self._loader(lbl_dir + 'kidney_left' + '.nii.gz')
            organ_lbl[array > 0] = 3
        if os.path.exists(lbl_dir + 'kidney_right' + '.nii.gz'):
            array, mata_infomation = self._loader(lbl_dir + 'kidney_right' + '.nii.gz')
            organ_lbl[array > 0] = 3
        if os.path.exists(lbl_dir + 'liver_tumor' + '.nii.gz'):
            array, mata_infomation = self._loader(lbl_dir + 'liver_tumor' + '.nii.gz')
            organ_lbl[array > 0] = 4
        if os.path.exists(lbl_dir + 'pancreas_tumor' + '.nii.gz'):
            array, mata_infomation = self._loader(lbl_dir + 'pancreas_tumor' + '.nii.gz')
            organ_lbl[array > 0] = 5
        if os.path.exists(lbl_dir + 'pancreas_tumor' + '.nii.gz'):
            array, mata_infomation = self._loader(lbl_dir + 'kidney_tumor' + '.nii.gz')
            organ_lbl[array > 0] = 6
        """

        if os.path.exists(lbl_dir + 'totalsegmentator' + '.nii.gz'):
            organ_lbl, mata_infomation = self._loader(lbl_dir + 'totalsegmentator' + '.nii.gz')

        return organ_lbl, mata_infomation

class LoadImageh5d(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        reader: Optional[Union[ImageReader, str]] = None,
        dtype: DtypeLike = np.float32,
        meta_keys: Optional[KeysCollection] = None,
        meta_key_postfix: str = DEFAULT_POST_FIX,
        overwriting: bool = False,
        image_only: bool = False,
        ensure_channel_first: bool = False,
        simple_keys: bool = False,
        allow_missing_keys: bool = False,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self._loader = LoadImage(reader, image_only, dtype, ensure_channel_first, simple_keys, *args, **kwargs)
        if not isinstance(meta_key_postfix, str):
            raise TypeError(f"meta_key_postfix must be a str but is {type(meta_key_postfix).__name__}.")
        self.meta_keys = ensure_tuple_rep(None, len(self.keys)) if meta_keys is None else ensure_tuple(meta_keys)
        if len(self.keys) != len(self.meta_keys):
            raise ValueError("meta_keys should have the same length as keys.")
        self.meta_key_postfix = ensure_tuple_rep(meta_key_postfix, len(self.keys))
        self.overwriting = overwriting


    def register(self, reader: ImageReader):
        self._loader.register(reader)


    def __call__(self, data, reader: Optional[ImageReader] = None):
        d = dict(data)
        # print('file_name', d['name'])
        for key, meta_key, meta_key_postfix in self.key_iterator(d, self.meta_keys, self.meta_key_postfix):
            data = self._loader(d[key], reader)
            if self._loader.image_only:
                d[key] = data
            else:
                if not isinstance(data, (tuple, list)):
                    raise ValueError("loader must return a tuple or list (because image_only=False was used).")
                d[key] = data[0]
                if not isinstance(data[1], dict):
                    raise ValueError("metadata must be a dict.")
                meta_key = meta_key or f"{key}_{meta_key_postfix}"
                if meta_key in d and not self.overwriting:
                    raise KeyError(f"Metadata with key {meta_key} already exists and overwriting=False.")
                d[meta_key] = data[1]

        #print(f"d is {d}")

        return d
    
def get_loader(args):
    train_transforms = Compose(
        [
            LoadImaged_BodyMap(keys=["ct", "pet"]),
            AddChanneld(keys=["ct", "pet", "label"]), # just trying to see if 2 channels work
            Orientationd(keys=["ct", "pet", "label"], axcodes="RAS"),
            Spacingd(
                keys=["ct", "pet", "label"],
                pixdim=(args.space_x, args.space_y, args.space_z),
                mode=("bilinear", "bilinear", "nearest"),
            ), # process h5 to here
            ScaleIntensityRanged(
                keys=["ct"],
                a_min=args.ct_a_min,
                a_max=args.ct_a_max,
                b_min=args.b_min,
                b_max=args.b_max,
                clip=True
            ),
            ScaleIntensityRanged(
                keys=["pet"],
                a_min=args.pet_a_min,
                a_max=args.pet_a_max,
                b_min=args.b_min,
                b_max=args.b_max,
                clip=True
            ),
            #ScaleIntensityRanged(
            #    keys=["image"],
            #    a_min=args.a_min,
            #    a_max=args.a_max,
            #    b_min=args.b_min,
            #    b_max=args.b_max,
            #    clip=True,
            #),
            ConcatItemsd(keys=["ct", "pet"], name="image", dim=0),
            SpatialPadd(keys=["image", "label"], spatial_size=(args.roi_x, args.roi_y, args.roi_z), mode=["minimum", "constant"]),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(args.roi_x, args.roi_y, args.roi_z), 
                pos=20,
                neg=1,
                num_samples=args.num_samples,
                image_key="image",
                image_threshold=-1,
            ),
            RandRotate90d(
                keys=["image", "label"],
                prob=0.10,
                max_k=3,
            ),
            ToTensord(keys=["image", "label"]),
        ]
    )

    val_transforms = Compose(
        [
            LoadImageh5d(keys=["ct", "pet"]), # just temporarily
            AddChanneld(keys=["ct", "pet", "label"]), # just trying to see if 2 channels work
            Orientationd(keys=["ct", "pet", "label"], axcodes="RAS"),
            Spacingd(
                keys=["ct", "pet", "label"],
                pixdim=(args.space_x, args.space_y, args.space_z),
                mode=("bilinear", "bilinear", "nearest"),
            ),
            ScaleIntensityRanged(
                keys=["ct"],
                a_min=args.ct_a_min,
                a_max=args.ct_a_max,
                b_min=args.b_min,
                b_max=args.b_max,
                clip=True
            ),
            ScaleIntensityRanged(
                keys=["pet"],
                a_min=args.pet_a_min,
                a_max=args.pet_a_max,
                b_min=args.b_min,
                b_max=args.b_max,
                clip=True
            ),
            #ScaleIntensityRanged(
            #    keys=["image"],
            #    a_min=args.a_min,
            #    a_max=args.a_max,
            #    b_min=args.b_min,
            #    b_max=args.b_max,
            #    clip=True,
            #),
            ConcatItemsd(keys=["ct", "pet"], name="image", dim=0),
            SpatialPadd(keys=["image", "label"], spatial_size=(args.roi_x, args.roi_y, args.roi_z), mode='constant'),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                pos=2,
                neg=0,
                num_samples=args.num_samples,
                image_key="image",
                image_threshold=-1,
            ),
            ToTensord(keys=["image", "label"]),
        ]
    )

    if args.phase == 'train':
        train_ct = []
        train_pet = []
        train_lbl = []
        train_name = []
        for line in open(os.path.join(args.data_txt_path, args.dataset_list + '.txt')):
            name = line.strip().split('\t')[0]
            train_ct.append(os.path.join(args.data_root_path, name + '/ct.nii.gz'))
            train_pet.append(os.path.join(args.data_root_path, name + '/pet.nii.gz'))
            train_lbl.append(os.path.join(args.data_root_path, name + '/segmentations/'))
            train_name.append(name)
        data_dicts_train = [{'ct': ct, 'pet': pet, 'label': label, 'name': name}
                            for ct, pet, label, name in zip(train_ct, train_pet, train_lbl, train_name)]
        print('train len {}'.format(len(data_dicts_train)))
        # data_dicts_train=data_dicts_train[:10]
        # breakpoint()

        if args.cache_dataset:
            if args.uniform_sample:
                train_dataset = UniformCacheDataset(data=data_dicts_train, transform=train_transforms,
                                                    cache_rate=args.cache_rate, datasetkey=args.datasetkey)
            else:
                train_dataset = CacheDataset(data=data_dicts_train, transform=train_transforms,
                                             cache_rate=args.cache_rate)
        else:
            if args.uniform_sample:
                train_dataset = UniformDataset(data=data_dicts_train, transform=train_transforms,
                                               datasetkey=args.datasetkey)
            else:
                train_dataset = Dataset(data=data_dicts_train, transform=train_transforms)

        if args.save_transform:
            train_dataset = DiskDataset(data=data_dicts_train, transform=train_transforms, save_dir=args.save_dir)
            train_sampler = DistributedSampler(dataset=train_dataset, even_divisible=True,
                                               shuffle=True) if args.dist else None
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
                                      num_workers=args.num_workers,
                                      collate_fn=list_data_collate, sampler=train_sampler) # should change collate back to
            # list_data_collate if you want to do batch size = 2, but trying pad_list_data_collate now
            return train_loader, train_sampler, len(train_dataset)

        train_sampler = DistributedSampler(dataset=train_dataset, even_divisible=True,
                                           shuffle=True) if args.dist else None
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
                                  num_workers=args.num_workers,
                                  collate_fn=list_data_collate, sampler=train_sampler)
        return train_loader, train_sampler, len(train_dataset)

    if args.phase == 'validation':
        val_ct = []
        val_pet = []
        val_lbl = []
        val_name = []
        for line in open(os.path.join(args.data_txt_path, args.dataset_val_list + '.txt')):
            name = line.strip().split('\t')[0]
            val_ct.append(os.path.join(args.data_root_path, name + '/ct.nii.gz'))
            val_pet.append(os.path.join(args.data_root_path, name + '/pet.nii.gz'))
            val_lbl.append(os.path.join(args.data_root_path, name + '/segmentations/'))
            val_name.append(name)
        # item in args.dataset_list:
        #    for line in open(os.path.join(args.data_txt_path,  item, 'real_tumor_val_0.txt')):
        #        name = line.strip().split()[1].split('.')[0]
        #        val_img.append(os.path.join(args.data_root_path, line.strip().split()[0]))
        #        val_lbl.append(os.path.join(args.data_root_path, line.strip().split()[1]))
        #        val_name.append(name)
        data_dicts_val = [{'ct': ct, 'pet': pet, 'label': label, 'name': name}
                    for ct, pet, label, name in zip(val_ct, val_pet, val_lbl, val_name)]
        print('val len {}'.format(len(data_dicts_val)))
    
        if args.cache_dataset:
            val_dataset = CacheDataset(data=data_dicts_val, transform=val_transforms, cache_rate=args.cache_rate)
        else:
            val_dataset = Dataset(data=data_dicts_val, transform=val_transforms)

        if args.save_transform:
            val_dataset = DiskDataset(data=data_dicts_val, transform=val_transforms, save_dir=args.save_dir_val)
            val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False,
                                      num_workers=args.num_workers,
                                      collate_fn=list_data_collate) # should change collate back to
            # list_data_collate if you want to do batch size = 2, but trying pad_list_data_collate now
            return val_loader, val_transforms, len(val_dataset)

        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=list_data_collate)
        return val_loader, val_transforms, len(val_dataset)
    

def get_key(name):
    ## input: name
    ## output: the corresponding key
    dataset_index = int(name[0:2])
    if dataset_index == 10:
        template_key = name[0:2] + '_' + name[17:19]
    else:
        template_key = name[0:2]
    return template_key

if __name__ == "__main__":
    """
    train_loader, test_loader = partial_label_dataloader()
    for index, item in enumerate(test_loader):
        print(item['image'].shape, item['label'].shape, item['task_id'])
        input()
    """


    class Config:
        def __init__(self):
            self.name = "synt"
            self.image_channels = 2
            self.dataset_list = "Step_1_datalist_train"
            self.data_root_path = "../../../DiffTumor_data/Autopet/imagesTr_Step_1_pet_ct/"
            self.data_txt_path = "../../../DiffTumor_data/Autopet/"
            self.dist = False
            self.batch_size = 1
            self.num_workers = 0
            self.ct_a_min = -832.062744140625
            self.ct_a_max = 1127.758544921875
            self.pet_a_min = 1.0433332920074463
            self.pet_a_max = 51.211158752441406
            self.b_min = -1.0
            self.b_max = 1.0
            self.space_x = 1.0
            self.space_y = 1.0
            self.space_z = 1.0
            self.roi_x = 96
            self.roi_y = 96
            self.roi_z = 96
            self.num_samples = 2
            self.phase = "validation"
            self.uniform_sample = False
            self.cache_dataset = False
            self.cache_rate = 0.027
            self.save_transform = True
            self.save_dir = "../../../DiffTumor_data/Autopet/imagesTr_Step_1_pet_ct_processed"
            self.dataset_val_list = "Step_1_datalist_val"
            self.save_dir_val = "../../../DiffTumor_data/Autopet/imagesTs_Step_1_pet_ct_processed"


    # Usage
    config = Config()
    print(config.name)  # Outputs: "synt"

    #print(config)

    val_loader, _, length = get_loader(config)

    from tqdm import tqdm

    for item in tqdm(val_loader, total=len(val_loader)):
        pass

