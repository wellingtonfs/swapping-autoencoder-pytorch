#from skimage.io import imread, imsave
#from skimage.transform import resize
#import sys

#sys.path.insert(0, "/home/wellington/Documentos/swapping-autoencoder-pytorch")

from shutil import rmtree, move
from util import ConnectionServer, unzip
import os

def find_novo_nome(path, formato):
    ini = len(os.listdir(path))
    while os.path.isfile(str(ini)+"."+formato):
        ini += 1

    return str(ini)+"."+formato

def download_data(tipo, qtd, server):
    trs = [None] * 10

    for i in range(qtd[0], qtd[1]):
        keep = True
        while keep:
            for j in range(len(trs)):
                if trs[j] is not None and trs[j].is_alive():
                    continue

                trs[j] = server.download(tipo, "%d.zip"%i, paralelo=True)

                keep = False
                break

    while any(trs[i].is_alive() for i in range(len(trs)) if trs[i] != None):
        continue

class Phoenix:
    @staticmethod
    def download(opt, tipo="train", split=[]):
        if split == []:
            split = [0,567] if tipo == "train" else [0,63]
            split = [0,55] if tipo == "dev" else split
        elif len(split) != 2:
            raise Exception("split tem que ser do formato: [ini, fim] ou []")

        split_str = str(split[0]) + '-' + str(split[1])

        #download dataset

        #tipo pode ser: train, dev ou test
        server = ConnectionServer(opt)

        download_data(tipo, split, server)

    @staticmethod
    def descompactar(opt, salto=1):
        assert salto > 0
        if not os.path.isdir(opt.dataroot):
            os.makedirs(opt.dataroot)

        for filezipname in [v for v in os.listdir(opt.download_dir) if ".zip" in v]:
            pathfilezip = os.path.join(opt.download_dir, filezipname)
            path_to = os.path.join(opt.download_dir, "unzip")
            unzip(pathfilezip, path_to)

            for filename in [v for i, v in enumerate(os.listdir(path_to)) if i % salto == 0]:
                pathfile = os.path.join(path_to, filename)
                move(pathfile, os.path.join(opt.dataroot, find_novo_nome(opt.dataroot, filename.split('.')[1])))

            rmtree(path_to)

'''
if __name__ == "__main__":
    class options:
        def __init__(self):
            self.ip = "http://179.189.133.252:3005"
            self.dataroot = "/home/wellington/Documentos/Geral/TesteSwapping/unzip"
            self.download_dir = "/home/wellington/Documentos/Geral/TesteSwapping/download"

    opt = options()

    Phoenix.download(opt, split=[0,5])
    Phoenix.descompactar2(opt)
'''