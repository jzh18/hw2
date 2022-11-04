import numpy as np
from .autograd import Tensor
import gzip

from typing import Iterator, Optional, List, Sized, Union, Iterable, Any


def parse_mnist(image_filesname, label_filename):
    """ Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded
                data.  The dimensionality of the data should be
                (num_examples x input_dim) where 'input_dim' is the full
                dimension of the data, e.g., since MNIST images are 28x28, it
                will be 784.  Values should be of type np.float32, and the data
                should be normalized to have a minimum value of 0.0 and a
                maximum value of 1.0.

            y (numpy.ndarray[dypte=np.int8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.int8 and
                for MNIST will contain the values 0-9.
    """
    # BEGIN YOUR SOLUTION
    byte_order = 'big'
    X = []
    with gzip.open(image_filesname) as f:
        magic_num = int.from_bytes(f.read(4), byteorder=byte_order)
        if magic_num != 2051:
            raise Exception('decode error!')
        num_of_imgs = int.from_bytes(f.read(4), byteorder=byte_order)
        num_of_rows = int.from_bytes(f.read(4), byteorder=byte_order)
        num_of_cols = int.from_bytes(f.read(4), byteorder=byte_order)
        for i in range(num_of_imgs):
            img_one_dim = np.frombuffer(
                f.read(num_of_rows * num_of_cols), dtype=np.uint8).astype(np.float32)
            X.append(img_one_dim)
    X = np.array(X)
    min = np.min(X)
    max = np.max(X)
    X = (X - min) / (max - min)
    y = []
    with gzip.open(label_filename) as f:
        magic_num = int.from_bytes(f.read(4), byteorder=byte_order)
        if magic_num != 2049:
            raise Exception('decode error!')
        num_of_imgs = int.from_bytes(f.read(4), byteorder=byte_order)
        for i in range(num_of_imgs):
            label = np.frombuffer(f.read(1), dtype=np.uint8)
            y.append(label)

    return X, np.array(y).squeeze()

    # END YOUR SOLUTION


class Transform:
    def __call__(self, x):
        raise NotImplementedError


class RandomFlipHorizontal(Transform):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        """
        Horizonally flip an image, specified as n H x W x C NDArray.
        Args:
            img: H x W x C NDArray of an image
        Returns:
            H x W x C ndarray corresponding to image flipped with probability self.p
        Note: use the provided code to provide randomness, for easier testing
        """
        flip_img = np.random.rand() < self.p
        # BEGIN YOUR SOLUTION
        if flip_img:
            return np.fliplr(img)
        else:
            return img
        # END YOUR SOLUTION


class RandomCrop(Transform):
    def __init__(self, padding=3):
        self.padding = padding

    def __call__(self, img):
        """ Zero pad and then randomly crop an image.
        Args:
             img: H x W x C NDArray of an image
        Return 
            H x W x C NAArray of cliped image
        Note: generate the image shifted by shift_x, shift_y specified below
        """
        # print(f'padding: {self.padding}')
        # print(f'original shape: {img.shape}')
        shift_x, shift_y = np.random.randint(
            low=-self.padding, high=self.padding+1, size=2)
        # BEGIN YOUR SOLUTION
        h, w, c = img.shape
        padding_img = np.zeros((h+2*self.padding, w+2*self.padding, c))
        # print(
        #     f'selected shape: {padding_img[self.padding:-self.padding,self.padding:-self.padding, :].shape}')
        if self.padding != 0:
            padding_img[self.padding:-self.padding,
                        self.padding:-self.padding, :] = img
        else:
            padding_img = img
        # print(f'padding img shape: {padding_img.shape}')
        # print(f'shift x: {shift_x}')
        # print(f'shift y: {shift_y}')

        shift_img = padding_img[self.padding+shift_x:(self.padding+h+shift_x),
                                self.padding+shift_y:(self.padding+w+shift_y), :]
        # print(f'shift img shape: {shift_img.shape}')
        return shift_img
        # END YOUR SOLUTION


class Dataset:
    r"""An abstract class representing a `Dataset`.

    All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given key. Subclasses must also overwrite
    :meth:`__len__`, which is expected to return the size of the dataset.
    """

    def __init__(self, transforms: Optional[List] = None):
        self.transforms = transforms

    def __getitem__(self, index) -> object:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def apply_transforms(self, x):
        if self.transforms is not None:
            # apply the transforms
            for tform in self.transforms:
                x = tform(x)
        return x


class DataLoader:
    r"""
    Data loader. Combines a dataset and a sampler, and provides an iterable over
    the given dataset.
    Args:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: ``1``).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
     """
    dataset: Dataset
    batch_size: Optional[int]

    def __init__(
        self,
        dataset: Dataset,
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
    ):

        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size
        if not self.shuffle:
            self.ordering = np.array_split(np.arange(len(dataset)),
                                           range(batch_size, len(dataset), batch_size))

    def __iter__(self):
        # BEGIN YOUR SOLUTION
        if self.shuffle:
            all_indices = np.arange(len(self.dataset))
            np.random.shuffle(all_indices)
            self.ordering = np.array_split(all_indices, range(
                self.batch_size, len(self.dataset), self.batch_size))
            self.ordering
        self.n = len(self.ordering)
        self.i = 0
        # END YOUR SOLUTION
        return self

    def __next__(self):
        # BEGIN YOUR SOLUTION
        if self.i >= self.n:
            raise StopIteration
        indices = self.ordering[self.i]
        self.i += 1
        
        all_res=[]
        for i in indices:
            data = self.dataset[i]
            for j,v in enumerate(data):
                if j+1>len(all_res):
                    all_res.append(list())
                all_res[j].append(v)
        
        results=[]
        for v in all_res:
            t=Tensor(v)
            results.append(t)
        return results
                
        
        
        
        # END YOUR SOLUTION


class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        # BEGIN YOUR SOLUTION
        super().__init__(transforms)
        X, y = parse_mnist(image_filename, label_filename)
        size = len(X)
        self.X = X.reshape(size, 28, 28, 1)
        self.y = y
        # END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        # BEGIN YOUR SOLUTION
        x, y = self.X[index], self.y[index]
        x = self.apply_transforms(x)
        return x, y
        # END YOUR SOLUTION

    def __len__(self) -> int:
        # BEGIN YOUR SOLUTION
        return len(self.X)
        # END YOUR SOLUTION


class NDArrayDataset(Dataset):
    def __init__(self, *arrays):
        self.arrays = arrays

    def __len__(self) -> int:
        return self.arrays[0].shape[0]

    def __getitem__(self, i) -> object:
        return tuple([a[i] for a in self.arrays])
