

import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from backbone.ResNet18 import resnet18
import torch.nn.functional as F
from utils.conf import base_path
from PIL import Image
import os
from datasets.utils.validation import get_train_val
from datasets.utils.continual_dataset import ContinualDataset, store_masked_loaders
from datasets.transforms.denormalization import DeNormalize


class TinyImagenet(Dataset):
    N_CLASSES = 200
    classes = ['egyptian_cat','reel','volleyball','rocking_chair','lemon','bullfrog','basketball','cliff','espresso','plunger','parking_meter','german_shepherd','dining_table','monarch','brown_bear','school_bus','pizza','guinea_pig','umbrella','organ','oboe','maypole','goldfish','potpie','hourglass','seashore','computer_keyboard','arabian_camel','ice_cream','nail','space_heater','cardigan','baboon','snail','coral_reef','albatross','spider_web','sea_cucumber','backpack','labrador_retriever','pretzel','king_penguin','sulphur_butterfly','tarantula','lesser_panda','pop_bottle','banana','sock','cockroach','projectile','beer_bottle','mantis','freight_car','guacamole','remote_control','european_fire_salamander','lakeside','chimpanzee','pay-phone','fur_coat','alp','lampshade','torch','abacus','moving_van','barrel','tabby','goose','koala','bullet_train','cd_player','teapot','birdhouse','gazelle','academic_gown','tractor','ladybug','miniskirt','golden_retriever','triumphal_arch','cannon','neck_brace','sombrero','gasmask','candle','desk','frying_pan','bee','dam','spiny_lobster','police_van','ipod','punching_bag','beacon','jellyfish','wok',"potter's_wheel",'sandal','pill_bottle','butcher_shop','slug','hog','cougar','crane','vestment','dragonfly','cash_machine','mushroom','jinrikisha','water_tower','chest','snorkel','sunglasses','fly','limousine','black_stork','dugong','sports_car','water_jug','suspension_bridge','ox','ice_lolly','turnstile','christmas_stocking','broom','scorpion','wooden_spoon','picket_fence','rugby_ball','sewing_machine','steel_arch_bridge','persian_cat','refrigerator','barn','apron','yorkshire_terrier','swimming_trunks','stopwatch','lawn_mower','thatch','fountain','black_widow','bikini','plate','teddy','barbershop','confectionery','beach_wagon','scoreboard','orange','flagpole','american_lobster','trolleybus','drumstick','dumbbell','brass','bow_tie','convertible','bighorn','orangutan','american_alligator','centipede','syringe','go-kart','brain_coral','sea_slug','cliff_dwelling','mashed_potato','viaduct','military_uniform','pomegranate','chain','kimono','comic_book','trilobite','bison','pole','boa_constrictor','poncho','bathtub','grasshopper','walking_stick','chihuahua','tailed_frog','lion','altar','obelisk','beaker','bell_pepper','bannister','bucket','magnetic_compass','meat_loaf','gondola','standard_poodle','acorn','lifeboat','binoculars','cauliflower','african_elephant']

    """
    Defines Tiny Imagenet as for the others pytorch datasets.
    """
    def __init__(self, root: str, train: bool=True, transform: transforms=None,
                target_transform: transforms=None, download: bool=False) -> None:
        self.not_aug_transform = transforms.Compose([transforms.ToTensor()])
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        if download:
            if os.path.isdir(root) and len(os.listdir(root)) > 0:
                print('Download not needed, files already on disk.')
            else:
                from onedrivedownloader import download
                ln = '<iframe src="https://onedrive.live.com/embed?cid=D3924A2D106E0039&resid=D3924A2D106E0039%21111&authkey=AJ9v0OmtrqOpxFA" width="98" height="120" frameborder="0" scrolling="no"></iframe>'
                print('Downloading dataset')
                download(ln, filename=os.path.join(root, 'tiny-imagenet-processed.zip'), unzip=True, unzip_path=root, clean=True)
                

        self.data = []
        for num in range(20):
            self.data.append(np.load(os.path.join(
                root, 'processed/x_%s_%02d.npy' %
                      ('train' if self.train else 'val', num+1))))
        self.data = np.concatenate(np.array(self.data))

        self.targets = []
        for num in range(20):
            self.targets.append(np.load(os.path.join(
                root, 'processed/y_%s_%02d.npy' %
                      ('train' if self.train else 'val', num+1))))
        self.targets = np.concatenate(np.array(self.targets))

    def __len__(self):
        return len(self.data)

    

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.uint8(255 * img))
        original_img = img.copy()

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if hasattr(self, 'logits'):
            return img, target, original_img, self.logits[index]

        return img, target


class MyTinyImagenet(TinyImagenet):
    """
    Defines Tiny Imagenet as for the others pytorch datasets.
    """
    def __init__(self, root: str, train: bool=True, transform: transforms=None,
                target_transform: transforms=None, download: bool=False) -> None:
        super(MyTinyImagenet, self).__init__(
            root, train, transform, target_transform, download)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.uint8(255 * img))
        original_img = img.copy()

        not_aug_img = self.not_aug_transform(original_img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if hasattr(self, 'logits'):
          return img, target, not_aug_img, self.logits[index]

        return img, target,  not_aug_img

def base_path():
    return "/tmp/mbosc/"

class SequentialTinyImagenet(ContinualDataset):

    NAME = 'seq-tinyimg'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 20
    N_TASKS = 10
    MEAN = (0.4802, 0.4480, 0.3975)
    STD  = (0.2770, 0.2691, 0.2821)
    TRANSFORM = transforms.Compose(
            [transforms.RandomCrop(64, padding=4),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize(MEAN,
                                  STD)])

    def get_data_loaders(self):
        transform = self.TRANSFORM

        test_transform = self.TEST_TRANSFORM if hasattr(self, 'TEST_TRANSFORM') else transforms.Compose(
            [transforms.ToTensor(), self.get_normalization_transform()])

        train_dataset = MyTinyImagenet(base_path() + 'TINYIMG',
                                 train=True, download=True, transform=transform)
        if self.args.validation:
            train_dataset, test_dataset = get_train_val(train_dataset,
                                                    test_transform, self.NAME)
        else:
            test_dataset = TinyImagenet(base_path() + 'TINYIMG',
                        train=False, download=True, transform=test_transform)

        train, test = store_masked_loaders(train_dataset, test_dataset, self)
        return train, test


    @staticmethod
    def get_backbone():
        return resnet18(SequentialTinyImagenet.N_CLASSES_PER_TASK
                        * SequentialTinyImagenet.N_TASKS)

    @staticmethod
    def get_loss():
        return F.cross_entropy

    def get_transform(self):
        transform = transforms.Compose(
            [transforms.ToPILImage(), self.TRANSFORM])
        return transform

    @staticmethod
    def get_normalization_transform():
        transform = transforms.Normalize(SequentialTinyImagenet.MEAN,
                                         SequentialTinyImagenet.STD )
        return transform

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize(SequentialTinyImagenet.MEAN,
                                SequentialTinyImagenet.STD )
        return transform

class SequentialTinyImagenet32(SequentialTinyImagenet):
    MEAN,STD = [0.4807, 0.4485, 0.3980],[0.2541, 0.2456, 0.2604]
    TRANSFORM = transforms.Compose(
            [
             transforms.RandomCrop(32, padding=4),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize(MEAN,
                                  STD)])
    TEST_TRANSFORM = transforms.Compose(
        [
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize(MEAN,
                                STD)])
    

class SequentialTinyImagenet32R(SequentialTinyImagenet):
    MEAN,STD = [0.4807, 0.4485, 0.3980],[0.2541, 0.2456, 0.2604]
    TRANSFORM = transforms.Compose(
            [
             transforms.Resize(32),
             transforms.RandomCrop(32, padding=4),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize(MEAN,
                                  STD)])
    TEST_TRANSFORM = transforms.Compose(
        [
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize(MEAN,
                                STD)])
    
