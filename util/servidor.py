from threading import Thread
from shutil import make_archive
import requests
import os, time

def download_file(opt, tipo, nome, path, cont=0):
    if not os.path.isdir(path):
        os.makedirs(path)

    try:
        resposta = requests.get(os.path.join(opt.ip, "download", tipo, nome), verify=False)
    except requests.exceptions.ConnectionError:
        if cont > 4:
            return False

        time.sleep(1)
            
        return download_file(opt, tipo, nome, path, cont=cont+1)

    if resposta.status_code == requests.codes.OK:
        with open(os.path.join(path, nome), 'wb') as novo_arquivo:
            novo_arquivo.write(resposta.content)

        return True
    return False

    
def enviar_arquivo(opt, tipo, path_file, cont=0):
    files = {'docfile': open(path_file, "rb")}
    try:
        r = requests.post(os.path.join(opt.ip, "upload", tipo), files=files)
    except requests.exceptions.ConnectionError:
        if cont > 4:
            print("Erro ao baixar arquivo:", nome)
            return False

        time.sleep(1)
        
        return enviar_arquivo(opt, tipo, path_file, cont=cont+1)

class download_paralelo(Thread):
    def __init__ (self, opt, tipo, nome):
        Thread.__init__(self)
        self.opt = opt
        self.tipo = tipo
        self.nome = nome

    def run(self):
        if not download_file(self.opt, self.tipo, self.nome, self.opt.download_dir):
            print("Problemas ao baixar o arquivo! '%s'"%self.nome)

class enviar_paralelamente(Thread):
    def __init__ (self, opt, tipo, path_file):
        Thread.__init__(self)
        self.opt = opt
        self.tipo = tipo
        self.path_file = path_file

    def run(self):
        enviar_arquivo(self.opt, self.tipo, self.path_file)

class ConnectionServer:
    def __init__(self, opt):
        self.opt = opt

    def download(self, tipo, nome, paralelo=False):
        if paralelo:
            mythread = download_paralelo(self.opt, tipo, nome)
            mythread.start()
            return mythread
        
        return download_file(self.opt, tipo, nome, self.opt.download_dir)

    def upload(self, tipo, path_file, paralelo=False):
        if paralelo:
            mythread = enviar_paralelamente(self.opt, tipo, path_file)
            mythread.start()
            return mythread
        enviar_arquivo(self.opt, tipo, path_file)
        return True

    def upload_folder(self, tipo, path_folder, paralelo=False):
        if not os.path.isdir(path_folder):
            return False

        path_to = os.path.dirname(path_folder)

        if not os.path.isdir(path_to):
            return False

        path_file = os.path.join(path_to, "up_"+os.path.basename(path_folder))

        if os.path.isfile(path_file+".zip"):
            os.remove(path_file+".zip")

        make_archive(path_file, "zip", path_folder)

        return self.upload(tipo, path_file+".zip", paralelo=paralelo)

'''
if __name__ == "__main__":
    class options:
        def __init__(self):
            self.ip = "http://179.189.133.252:3005"
            self.dataroot = "/home/wellington/Documentos/Geral/TesteSwapping/unzip"
            self.download_dir = "/home/wellington/Documentos/Geral/TesteSwapping/download"

    opt = options()

    c = ConnectionServer(opt)
    c.upload_folder("var", "/home/wellington/Documentos/swapping-autoencoder-pytorch/data")
'''