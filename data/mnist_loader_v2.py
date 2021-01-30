import random

import numpy as np
import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class MNISTDataset():
    def __init__(self, transform):
        self.transform = transform
        mnist_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=None)
        self.data = mnist_dataset.data
        self.targets = mnist_dataset.targets
        
    def __len__(self):
        #print(len(self.annotations))
        return len(self.data)
    def __getitem__(self, index):
        images = []
        targets = []
        bboxes = []
        rotations = []
        wx, hy = [], []
        shifts = []
        offsets = []
        for i in range(9):  # adjust according to grid size
            index = random.randint(0, self.__len__() - 1)
            img, target = self.data[index], int(self.targets[index])

            img = Image.fromarray(img.numpy(), mode='L')
            if self.transform is not None:
                # TODO: Train LostGAN simply with MNIST
                x, y = random.randint(28, 28), random.randint(28, 28)
                img = img.resize((x, y), Image.ANTIALIAS)
                shift = random.randint(0, 0)
#                 rotation = random.choice([0, 90, -90, 180])
#                 img = img.rotate(rotation)
#                 rotations.append(rotation)

                img_w, img_h = img.size
                new_im = Image.new('L', (43, 43), 0)  # 32 to simulate padding and resulting in 128x128
                bg_w, bg_h = new_im.size
                offset = ((bg_w - img_w) // 2, (bg_h - img_h) // 2)
                # new_im=new_im.transform(size=(32, 32), method=Image.ANTIALIAS, fillcolor='red')
                new_im.paste(img, offset)  # , (int((32 - x) / 2), int((32 - y) / 2)))
                new_im = shift_2d(new_im, shift)
                new_im = Image.fromarray(new_im)
                # new_im.save(mnist_dir + '\custom_mnist\ctry\sav.png')
                # WARNING: Bounding boxes do not correspond to the ones in Multi-MNIST Think of the digit '1',
                # a bounding box with width = height = 28 is not 'correct' See for MultiMNIST,
                # https://github.com/aakhundov/tf-attend-infer-repeat FIXME: bounding boxes need to adapted depending
                #  on the rotation
                # bboxes.append([(32 - x) / 2 + 32 * i, (32 - y) / 2 + 32 * i, x, y])
                # WARNING: Not 100% sure about this calculation bboxes.append(new_im.getbbox())
                img = self.transform(new_im)
                wx.append(x)
                hy.append(y)
                shifts.append(shift)
                offsets.append(offset)
            images.append(img)

            # if self.target_transform is not None:
                # target = self.target_transform(target)
            targets.append(target)

        bboxes = make_bbox(wx, hy, shifts, offsets, grid_size=3)
        img_grid = torchvision.utils.make_grid(images, nrow=3, padding=0)
        targets = torch.tensor(np.array(targets), dtype=torch.int8)
        # bboxes = torch.tensor(np.array(bboxes), dtype=torch.int32)
        bboxes = torch.tensor(bboxes, dtype=torch.int32)
        bboxes = torch.div(bboxes, torch.Tensor([128, 128, 128, 128]))
#         rotations = torch.tensor(np.array(rotations))

        resize_grid = transforms.Compose([
                            transforms.ToPILImage(),
                            transforms.Resize((128, 128)),
                            transforms.ToTensor()])
        img_grid = resize_grid(img_grid)

        return img_grid, targets, bboxes  # , rotations


def shift_2d(image, shift):
    # print(shift)
    max_shift = 25
    # max_shift += 1
    #    shifted_images=[]
    # for image, shift in zip(images, shifts):
    padded_image = np.pad(image, max_shift, 'constant')
    rolled_image = np.roll(padded_image, shift, axis=0)
    rolled_image = np.roll(rolled_image, shift, axis=1)
    shifted_image = rolled_image[max_shift:-max_shift, max_shift:-max_shift]
    #    shifted_images.append(shifted_image)
    return shifted_image


def make_bbox(wx, hy, shifts, offsets, grid_size=2):
    bbox = []
    scaled_bbox = []
    w, h = 42, 42

    for x, y, shift, offset in zip(wx, hy, shifts, offsets):
        scaled_bbox.append(np.array([0+ offset[0], 0+ offset[1], x + shift, y + shift]))
    # print(np.vstack(scaled_bbox))

    for i in range(grid_size):
        for j in range(grid_size):
            bbox.append(np.array([j * w, i * h, 0, 0]))
    return np.vstack([(sum(x)) for x in zip(scaled_bbox, bbox)])

# mnist_dir = 'D:\_Personal Files\Studium\Thesis\controlling-gans\LostGANs'
# ## mnist_dir = '/netscratch/asharma/ds/mnist/'
#
# transform = transforms.Compose([
#     #    transforms.Resize((80, 80)),
#     transforms.ToTensor()
#     #    transforms.Normalize((0.5,), (0.5,))
# ])
#
# train_loader = DataLoader(
#     MNISTDataset(mnist_dir + '\data', train=True, download=False, transform=transform),
#     batch_size=4, shuffle=False)
#
# for i, data in enumerate(train_loader):
#     #    save_image(images, mnist_dir + '\custom_mnist\images_1x1_test\{idx}.png'.format(idx=i))
#     # images, labels, bbox = make_grid(data)
#     # print(images.size(), labels)
#     # save_image(images,mnist_dir+'\custom_mnist\ctry', nrow=2)
#     img, targets, bboxes, rotations = data
#     rotations = rotations.to(torch.int64)
#     # print(rotations)
#     # rot = torch.nn.functional.one_hot(rotations, 181)
#     em = torch.nn.Embedding(185, 1)
#     rot = em(rotations)
#
#     to_pil = transforms.ToPILImage(mode='RGB')
#     image = to_pil(img[0])
#     # figure, ax = plt.subplots(1)
#     fig = plt.figure()
#     ax = fig.add_subplot(111, aspect='equal')
#     rect = []
#     for i in range(16):
#         x = np.asarray(bboxes[0][i][0]).__int__()
#         y = np.asarray(bboxes[0][i][1]).__int__()
#         w = np.asarray(bboxes[0][i][2]).__int__()
#         h = np.asarray(bboxes[0][i][3]).__int__()
#         # print(rotations[1][i])
#         rect.append(patches.Rectangle((x, y), w, h,
#                                       # angle=rotations[1][i],
#                                       edgecolor='r',
#                                       alpha=1,
#                                       fill=False))  # stupid!
#     ax.imshow(image)
#     #    ax.add_patch(rect)
#     ax.add_collection(PatchCollection(rect, fc='none', ec='red'))
#     plt.savefig(mnist_dir + '\custom_mnist\ctry\imfig.png')
#     save_image(img[0], mnist_dir + '\custom_mnist\ctry\im3.png')
#     break
