import random
from data.image_folder import make_dataset
from PIL import Image
from skimage.io import imread

from data.base_dataset import BaseDataset, get_transform
from shutil import rmtree, move
from util import ConnectionServer, unzip, change, make_pose
import os, json, torch

class PhoenixDataset(BaseDataset):
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)

        self.opt = opt

        if self.opt.split == []:
            self.opt.split = [0,567] if "train" in self.opt.tipo_dados else [0,63]
            self.opt.split = [0,55] if "dev" in self.opt.tipo_dados else self.opt.split
        elif len(self.opt.split) != 2:
            raise Exception("opt.split tem que ser do formato: [ini, fim] ou []")

        split_str = str(self.opt.split[0]) + '-' + str(self.opt.split[1])

        #download dataset e poses
        dirs = {
            "256_train": ["/content/raw_256train_"+split_str, "/content/raw_256train_poses_"+split_str],
            "train": ["/content/raw_train_"+split_str, "/content/raw_train_poses_"+split_str],
            "dev": ["/content/raw_dev_"+split_str, "/content/raw_dev_poses_"+split_str],
            "test": ["/content/raw_test_"+split_str, "/content/raw_test_poses_"+split_str]
        }
        
        for path in dirs[self.opt.tipo_dados]:
            if os.path.isdir(path):
                rmtree(path)
            os.makedirs(path)

        #tipo pode ser: train, dev ou test
        print("iniciando!")
        self.__download_data(dirs[self.opt.tipo_dados])

        print("\ndescompactando")
        self.__unzip(dirs[self.opt.tipo_dados])
        
        self.data = []
        for dirname, drs, fls in os.walk(dirs[self.opt.tipo_dados][0]):
            i, t = 0, 1
            for fl in fls:
                if ".zip" in fl:
                    continue

                if i % t != 0:
                    i += 1
                    continue

                self.data.append(
                    [
                        os.path.join(dirname, fl),
                        change(dirs[self.opt.tipo_dados][0], dirs[self.opt.tipo_dados][1], '.'.join(os.path.join(dirname, fl).split(".")[:-1]) + "_keypoints.json")
                    ]
                )
                
                i += 1

        self.size_data = len(self.data)

        self.transform = get_transform(self.opt, grayscale=False)

    def __download_data(self, paths):
        server = ConnectionServer(self.opt)

        trs = [None] * 10

        for i in range(self.opt.split[0], self.opt.split[1]):
            keep, pose = True, False
            while keep:
                for j in range(len(trs)):
                    if trs[j] is not None and trs[j].is_alive():
                        continue

                    if pose:
                        tipo = "train" if "train" in self.opt.tipo_dados else "dev"
                        tipo = "test" if "test" in self.opt.tipo_dados else tipo

                        trs[j] = server.download(tipo, "%d_poses.zip"%i, paralelo=True, path_to=paths[1])

                        keep = False
                        break
                    else:
                        trs[j] = server.download(self.opt.tipo_dados, "%d.zip"%i, paralelo=True, path_to=paths[0])

                        pose = True
        
            if i % 10 == 0:
                print(str(i), end="")
            else:
                print(".", end="")

            if i % 100 == 0 and i != 0:
                print("")

        print("")
        while any(trs[i].is_alive() for i in range(len(trs)) if trs[i] != None):
            continue

    def __unzip(self, paths):
        #unzip imagens
        for file in os.listdir(paths[0]):
            path_file = os.path.join(paths[0], file)
            unzip(path_file, '.'.join(path_file.split('.')[:-1]))
            os.remove(path_file)

        #unzip poses
        for file in os.listdir(paths[1]):
            path_file = os.path.join(paths[1], file)
            unzip(path_file, "_poses.".join(path_file.split("_poses.")[:-1]))
            os.remove(path_file)

    def __len__(self):
        return self.size_data

    def __getitem__(self, idx):
        #if torch.is_tensor(idx):
        #    idx = idx.tolist()

        path_img, path_pose = self.data[idx % self.size_data]

        try:
            img = Image.open(path_img).convert('RGB')
            #img = imread(path_img)
        except OSError as err:
            print(err)
            return self.__getitem__(random.randint(0, len(self) - 1))

        with open(path_pose, "r") as f:
            pose = make_pose(
                self.opt.crop_size,
                json.load(f),
                tam_gauss=[self.opt.tam_gauss_menor, self.opt.tam_gauss_maior]
            )

        Img = self.transform(img)

        return {"real_A": Img, "path_A": path_img, "pose": pose}