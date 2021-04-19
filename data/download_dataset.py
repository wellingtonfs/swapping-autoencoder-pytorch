import sys

sys.path.insert(0, "/home/wellington/Documentos/swapping-autoencoder-pytorch")

import util, os
from shutil import rmtree

def download_data(self, tipo, qtd, paths, server):
    trs = [None] * 10

    for i in range(qtd[0], qtd[1]):
        keep = True
        while keep:
            for j in range(len(trs)):
                if trs[j] is not None and trs[j].isAlive():
                    continue

                trs[j] = server.download(self, tipo, nome, paralelo=True)

                keep = False
                break

    while any(trs[i].isAlive() for i in range(len(trs)) if trs[i] != None):
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
        dirs = {
            "train": "/content/raw_train_"+split_str,
            "dev": "/content/raw_dev_"+split_str,
            "test": "/content/raw_test_"+split_str
        }
        
        for path in dirs[tipo]:
            if os.path.isdir(path):
                rmtree(path)
            os.makedirs(path)

        #tipo pode ser: train, dev ou test
        print("iniciando!")

        server = ConnectionServer(opt)

        download_data(tipo, split, dirs[tipo], server)


if __name__ == "__main__":
    class options:
        def __init__(self):
            self.download_dir = "/home/wellington/Documentos/Geral/TesteSwapping/download"
            self.ip = "http://179.189.133.252:3005"
            self.dataroot = "/home/wellington/Documentos/Geral/TesteSwapping/unzip"

    opt = options()

    Phoenix.download(opt, split=[0,10])