import torch.utils.data as data

from PIL import Image, ImageTk
import os
import os.path
import six

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f).convert('RGB')
        return img

def default_loader(path):
    return pil_loader(path)

def default_flist_reader(flist, transform=None):
    """
    flist format: impath label\nimpath label\n ...(same to caffe's filelist)
    """
    imlist = []
    with open(flist, 'r') as rf:
        for line in rf.readlines():
            impath, imlabel = line.strip().rsplit(' ', 1)
            print(impath)
            im = default_loader(impath)

            if transform is not None:
                im = transform(im)

            imlist.append( (im, int(imlabel)) )
    
    return imlist

class ImageFilelist(data.Dataset):
    def __init__(self, root, flist, transform=None, target_transform=None,
            flist_reader=default_flist_reader, loader=default_loader):
        self.root   = root
        self.imlist = flist_reader(flist, transform=transform)        
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        img, target = self.imlist[index]
                
        return img, target

    def __len__(self):
        return len(self.imlist)