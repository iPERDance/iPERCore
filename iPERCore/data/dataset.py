from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os
import numpy as np

from .transforms import ToTensor, ImageTransformer


class DatasetFactory(object):
    def __init__(self):
        pass

    @staticmethod
    def get_by_name(dataset_name, opt, is_for_train):
        if dataset_name == "iPER":
            from .iPER_dataset import IPERDataset
            dataset = IPERDataset(opt, is_for_train)

        elif dataset_name == "iPERSeq":
            from .iPER_dataset import SeqIPERDataset
            dataset = SeqIPERDataset(opt, is_for_train)

        elif dataset_name == "fashion":
            from .fashion_dataset import FashionVideoDataset
            dataset = FashionVideoDataset(opt, is_for_train)

        elif dataset_name == "fashionSeq":
            from .fashion_dataset import SeqFashionVideoDataset
            dataset = SeqFashionVideoDataset(opt, is_for_train)

        elif dataset_name == "Seq_Concat_Place2":
            from .all_dataset import ConcatVideoPlaceDataset
            dataset = ConcatVideoPlaceDataset(opt, is_for_train)

        else:
            raise ValueError(f"Dataset {dataset_name} not recognized.")

        print(f"Dataset {dataset.name} was created.")
        return dataset


class DatasetBase(Dataset):
    def __init__(self, opt, is_for_train):
        super(DatasetBase, self).__init__()
        self._name = "BaseDataset"
        self._root = None
        self._opt = opt
        self._is_for_train = is_for_train
        self._intervals = opt.intervals
        self._create_transform()

        self._IMG_EXTENSIONS = [
            ".jpg", ".JPG", ".jpeg", ".JPEG",
            ".png", ".PNG", ".ppm", ".PPM", ".bmp", ".BMP",
        ]

    @property
    def name(self):
        return self._name

    @property
    def path(self):
        return self._root

    def _create_transform(self):
        self._transform = transforms.Compose([])

    def get_transform(self):
        return self._transform

    def _is_image_file(self, filename):
        return any(filename.endswith(extension) for extension in self._IMG_EXTENSIONS)

    def _is_csv_file(self, filename):
        return filename.endswith(".csv")

    def _get_all_files_in_subfolders(self, dir, is_file):
        images = []
        assert os.path.isdir(dir), "%s is not a valid directory" % dir

        for root, _, fnames in sorted(os.walk(dir)):
            for fname in fnames:
                if is_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)

        return images

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, item):
        raise NotImplementedError


class VideoDataset(DatasetBase):

    def __init__(self, opt, is_for_train):
        super(VideoDataset, self).__init__(opt, is_for_train)
        self._name = "VideoDataset"

        self._intervals = opt.intervals

        # read dataset
        self._read_dataset_paths()

    def __len__(self):
        return self._dataset_size

    def __getitem__(self, index):
        # assert (index < self._dataset_size)

        # start_time = time.time()
        # get sample data
        v_info = self._vids_info[index % self._num_videos]
        images, smpls = self._load_pairs(v_info)

        # pack data
        sample = {
            "images": images,
            "smpls": smpls,
            "offsets": np.zeros((1, 6890, 3), dtype=np.float32),
            "bg_ks": self._opt.bg_ks
        }

        sample = self._transform(sample)

        return sample

    def _read_dataset_paths(self):
        pass

    def _read_vids_info(self, file_path):
        pass

    @property
    def video_info(self):
        return self._vids_info

    def _load_pairs(self, vid_info):
        pass

    def _create_transform(self):
        transform_list = [
            ImageTransformer(output_size=self._opt.image_size),
            ToTensor()]
        self._transform = transforms.Compose(transform_list)
