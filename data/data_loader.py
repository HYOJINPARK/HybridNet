import os
from torch.utils import data
from torchvision import transforms
from PIL import Image
import torch

class Cifar_L:
    def __init__(self, data_set, transform=None):
        self.data_set = data_set
        if not None:
            self.transform = transform
        else:
            self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, index):

          image_name = self.data_set[index]
          label = image_name[1]
          image = Image.open(image_name[0]).convert('RGB')
          if self.transform:
              image1, image2= self.transform(image)
          else:
              image1 = image
              image2 = image

          return (image1, image2, label)

class Cifar_UL:
    def __init__(self, img_set, transform=None):
        self.img_set = img_set
        if not None:
            self.transform = transform
        else:
            self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.img_set)

    def __getitem__(self, index):

        image_name = self.img_set[index]
        image = Image.open(image_name).convert('RGB')
        if self.transform:
            image1, image2 = self.transform(image)
        else:
            image1 = image
            image2 = image

        return (image1, image2)

#
# def get_loader_spectro(image_path, image_size, batch_size, num_workers, rand, agument=False):
#     """Builds and returns Dataloader."""
#
#     if agument:
#         transform = transforms.Compose([
#             transforms.Scale(image_size+16),
#             transforms.RandomSizedCrop(image_size),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
#     else:
#         transform = transforms.Compose([
#             transforms.Scale(image_size),
#             transforms.ToTensor(),
#             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
#
#     dataset = ImageFolder(image_path, transform)
#     dataset.imgs = sorted(dataset.imgs, key=lambda x:x[0])
#     # class_sample_count = [15000, 15000]  # dataset has 10 class-1 samples, 1 class-2 samples, etc.
#     # weights = 1 / torch.Tensor(class_sample_count)
#     # weights = weights.double()
#     # sampler = data.sampler.WeightedRandomSampler(weights, batch_size)
#
#     if batch_size is None:
#         batch_size = dataset.__len__()
#
#     data_loader = data.DataLoader(dataset=dataset,
#                                   batch_size=batch_size,
#                                   shuffle=rand,
#                                   drop_last=True,
#                                   num_workers=num_workers)
#     return data_loader
#
#
# class SoundPicLoader:
#     def __init__(self, image_path, image_size, batch_size, lable_rate=0.5, label_num = 4, num_workers=4):
#         self.label_num = label_num
#         self.image_size = image_size
#
#         self.l_size = int(batch_size * lable_rate)
#         self.un_l_size = batch_size - self.l_size
#         self.spectro_size = batch_size #use whole batch
#         self.c_dim = 128#image_size
#
#         # label_name = ['Folk', 'Hiphop'] #, 'Pop', 'Rock']
#
#         self.unlabeled_loader = get_loader_spectro(image_path+'/unlabeled', image_size, self.un_l_size, num_workers, rand=True)
#         # self.labeled = [get_loader_spectro(image_path+'/labeled/'+label_name[n], image_size, self.l_size, num_workers, rand=True) for n in range(label_num)]
#         # self.sound = [get_loader_spectro(image_path+'/spectro/'+label_name[n], self.c_dim , self.spectro_size, num_workers, rand=True) for n in range(label_num)]
#         #
#         # # self.iter_unlabeled = iter(self.unlabeled)
#         # self.iter_labeled = [ iter(self.labeled[n]) for n in range(label_num)]
#         # self.iter_sound = [iter(self.sound[n]) for n in range(label_num)]
#
#         self.labeled = get_loader_spectro(image_path + '/labeled/',
#                                           image_size, self.l_size, num_workers,rand=True)
#         self.sound = get_loader_spectro(image_path + '/spectro/',
#                                         self.c_dim, self.spectro_size, num_workers, rand=True, agument=False)
#
#         # self.iter_unlabeled = iter(self.unlabeled)
#         self.iter_labeled = iter(self.labeled)
#         self.iter_sound = iter(self.sound)
#
#     def __iter__(self): #separate labeled and unlabeld
#         for idx, (x, y) in enumerate(self.unlabeled_loader):
#             y[:] = self.label_num
#
#             try:  # need to catch exception because it has less data
#                 l_x, l_y = next(self.iter_labeled)
#             except StopIteration:
#                 self.iter_labeled = iter(self.labeled)
#                 l_x, l_y = next(self.iter_labeled)
#                 pass
#             try:
#                 c, cy = next(self.iter_sound)
#             except StopIteration:
#                 self.iter_sound = iter(self.sound)
#                 c, cy = next(self.iter_sound)
#                 pass
#
#             x = torch.cat([x, l_x], 0)
#             y = torch.cat([y, l_y], 0)
#
#             yield x, y, c, cy
#
#     # def __iter__(self): #weighted version
#     #     for idx, (x, y) in enumerate(self.unlabeled_loader):
#     #         y[:] = self.label_num
#     #
#     #         # noise for unlabeled image
#     #         # shape = torch.Size((self.un_l_size, 3, self.c_dim, self.c_dim ))
#     #         # z = torch.cuda.FloatTensor(shape)
#     #         # torch.randn(shape, out=z)
#     #         # z = torch.cuda.FloatTensor(self.un_l_size, 3, self.c_dim, self.c_dim).normal_()
#     #
#     #         #refine later
#     #         l_x = [0] * self.label_num
#     #         l_y = [0] * self.label_num
#     #         l_c = [0] * self.label_num
#     #         l_cy = [0] * self.label_num
#     #         try : #need to catch exception because it has less data
#     #             l_x[0], l_y[0] = next(self.iter_labeled[0])
#     #             l_x[1], l_y[1] = next(self.iter_labeled[1])
#     #         except StopIteration:
#     #             self.iter_labeled = [iter(self.labeled[n]) for n in range(self.label_num)]
#     #             l_x[0], l_y[0] = next(self.iter_labeled[0])
#     #             l_x[1], l_y[1] = next(self.iter_labeled[1])
#     #             pass
#     #         try:
#     #             l_c[0], l_cy[0] = next(self.iter_sound[0])
#     #             l_c[1], l_cy[1] = next(self.iter_sound[1])
#     #         except StopIteration:
#     #             self.iter_sound = [iter(self.sound[n]) for n in range(self.label_num)]
#     #             l_c[0], l_cy[0] = next(self.iter_sound[0])
#     #             l_c[1], l_cy[1] = next(self.iter_sound[1])
#     #             pass
#     #
#     #         l_y[0][:] = 0
#     #         l_y[1][:] = 1
#     #         l_cy[0][:] = 0
#     #         l_cy[1][:] = 1
#     #
#     #         x = torch.cat([x, l_x[0], l_x[1]], 0)
#     #         y = torch.cat([y, l_y[0], l_y[1]], 0)
#     #         c = torch.cat([l_c[0], l_c[1]], 0)
#     #         cy = torch.cat([l_cy[0], l_cy[1]], 0)
#     #
#     #         yield x, y, c, cy
#
#
#     # def __next__(self):
#     #     if self.index >= self.size:
#     #         raise StopIteration
#     #
#     #     n = self.data[self.index]
#     #     self.index += 1
#     #     return n
#
#
# class SoundPicLoaderLabeled:
#     def __init__(self, image_path, image_size, batch_size, label_num = 4, num_workers=4):
#         self.label_num = label_num
#         self.image_size = image_size
#
#         self.l_size = batch_size
#         self.un_l_size = 64
#         self.spectro_size = batch_size #use whole batch
#         self.c_dim = image_size
#
#         self.unlabeled_loader = get_loader_spectro(image_path + '/unlabeled', image_size, self.un_l_size, num_workers,
#                                                    rand=True)
#         self.labeled = get_loader_spectro(image_path + '/labeled/',
#                                           image_size, self.l_size, num_workers,rand=True)
#         self.sound = get_loader_spectro(image_path + '/spectro/',
#                                         self.c_dim, self.spectro_size, num_workers, rand=True)
#
#         # self.iter_unlabeled = iter(self.unlabeled)
#         self.iter_labeled = iter(self.labeled)
#         self.iter_sound = iter(self.sound)
#
#     def __iter__(self): #separate labeled and unlabeld
#         for idx, (x, y) in enumerate(self.unlabeled_loader): #dummy
#
#             try:  # need to catch exception because it has less data
#                 x, y = next(self.iter_labeled)
#             except StopIteration:
#                 self.iter_labeled = iter(self.labeled)
#                 x, y = next(self.iter_labeled)
#                 pass
#
#
#             try:
#                 c, cy = next(self.iter_sound)
#             except StopIteration:
#                 self.iter_sound = iter(self.sound)
#                 c, cy = next(self.iter_sound)
#                 pass
#
#             yield x, y, c, cy