import matplotlib.pyplot as plt
import numpy as np
import zipfile, os
import torch

def unzip(path_file, to_dir):            
    with zipfile.ZipFile(path_file) as zf:
        zf.extractall(to_dir)

    return True

def change(path1, path2, path_root):
    p1 = [v for v in path1.split("/") if v != ""]
    p2 = [v for v in path2.split("/") if v != ""]
    pr = [v for v in path_root.split("/") if v != ""]

    pr = pr[len(p1):]
    new_path = []
    new_path.extend(p2)
    new_path.extend(pr)

    new_path = os.path.join(*new_path)
    if not os.path.isabs(new_path) and len(path2) > 0 and path2[0] == '/':
        new_path = "/" + new_path

    return new_path

def imshow(img):
    if img.max() > 1.0:
        img = img / 255.0

    if type(img) == np.ndarray:
        if len(img.shape) == 3:
            plt.imshow(img)
            plt.show()
        else:
            plt.imshow(img, cmap='gray')
            plt.show()
    else:
        with torch.no_grad():
            img = img.clone()
            img = img.to("cpu")

            if img.min() < 0.0: # "des"normalizar
                img = img / 2 + 0.5 # img * std + med,

            npimg = img.detach().numpy()
            if len(npimg.shape) == 3:
                plt.imshow(np.transpose(npimg, (1, 2, 0)))
                plt.show()
            else:
                plt.imshow(npimg, cmap='gray')
                plt.show()

def make_gauss(img2d, center=(32,32), amplitude=1.0, std=(1.0,1.0)):
    def pixel_value(x, y):
        cX, cY = center
        A = torch.tensor(amplitude)
        varX, varY = std

        res = torch.exp(torch.tensor(-1* (((x-cX)**2/float(2*varX)) + ((y-cY)**2/float(2*varY))))) * A
        return res

    for i in range(img2d.shape[0]):
        for j in range(img2d.shape[1]):
            img2d[i,j] = pixel_value(j, i) #invertido para que funcione corretamente o eixo x e y

#essa classe gera uma lista de camadas de gaussianas de forma mais rapida
class make_gauss_phoenix:
    def __init__(self, size_img=(), tam_gauss=[32, 64]):
        assert len(size_img) == 2 and len(tam_gauss) == 2
        self.size_img = size_img
        self.tam_gauss = tam_gauss

        num_to_par = 6 if tam_gauss[0] % 2 == 0 else 7
        self.tam_gauss[0] += num_to_par

        #gerar uma gaussiana menor
        self.gauss = torch.zeros(self.tam_gauss[0], self.tam_gauss[0])
        make_gauss(
            self.gauss,
            center=(self.tam_gauss[0]//2,self.tam_gauss[0]//2),
            amplitude=1,
            std=(
                self.tam_gauss[0]-num_to_par,
                self.tam_gauss[0]-num_to_par
            )
        )

        num_to_par = 6 if tam_gauss[1] % 2 == 0 else 7
        self.tam_gauss[1] += num_to_par

        #gerar uma gaussiana maior
        self.gauss_maior = torch.zeros(self.tam_gauss[1], self.tam_gauss[1])

        make_gauss(
            self.gauss_maior,
            center=(self.tam_gauss[1]//2, self.tam_gauss[1]//2),
            amplitude=1,
            std=(self.tam_gauss[1]-num_to_par, self.tam_gauss[1]-num_to_par)
        )

    def make(self, list_center, maior=False):
        assert len(list_center) > 0

        itr = 1 if maior else 0

        #criar imagem de saida
        img_full = torch.zeros(self.size_img[0]+self.tam_gauss[itr], self.size_img[1]+self.tam_gauss[itr])

        #rodar todos os pontos que deve ter uma gaussiana
        for ponto in list_center:
            x, y = ponto

            #como img_full é maior que a imagem de saida, temos que fixar o ponto 0,0 da imagem menor com
            #a compensação da imagem maior
            center = (x + self.tam_gauss[itr]//2, y + self.tam_gauss[itr]//2)

            xi, xf = center[0] - self.tam_gauss[itr]//2, center[0] + self.tam_gauss[itr]//2
            yi, yf = center[1] - self.tam_gauss[itr]//2, center[1] + self.tam_gauss[itr]//2

            img_full[xi:xf, yi:yf] += self.gauss_maior if maior else self.gauss

        return torch.transpose(
            img_full[self.tam_gauss[itr]//2:-self.tam_gauss[itr]//2, self.tam_gauss[itr]//2:-self.tam_gauss[itr]//2],
                1, 0).clamp_(0.0, 1.0)

#converte um ponto x, y de uma dimenção de imagem para outra
def convert_point_to_dim(ponto, size):
    xx = int((ponto[0] / 260.0) * size)
    yy = int((ponto[1] / 260.0) * size)

    xx = xx if xx < size else size - 1
    yy = yy if yy < size else size - 1
    x = xx if xx >= 0 else 0
    y = yy if yy >= 0 else 0
    return x, y


#essa função auxilia na função make_pose
#dado uma lista de pontos, gera uma saida de pontos gaussiano representando a pose
def make_gauss_matriz(infos_to_make, all_pontos, tipo, tam_gauss=[32,160]):
    size = infos_to_make["size"]
    local_pose = torch.zeros(1, size, size)
    all_pontos = torch.tensor(all_pontos).view(-1, 3)

    if tipo == "pose":
        ignorar_pontos = [1, 2, 3, 5, 6, 8]

        local_pose[0] = infos_to_make["make"].make([
            convert_point_to_dim(
                (all_pontos[i][0], all_pontos[i][1]), infos_to_make["size"]
            ) for i in infos_to_make["pose"] if not i in ignorar_pontos
        ], maior=True)

        local_pose[0] += infos_to_make["make"].make([
            convert_point_to_dim(
                (all_pontos[i][0], all_pontos[i][1]), infos_to_make["size"]
            ) for i in ignorar_pontos
        ])

    elif tipo == "hand" or tipo == "face":
        for i in infos_to_make[tipo]:
            local_pose[-1] = infos_to_make["make"].make([
                convert_point_to_dim(
                    (all_pontos[i][0], all_pontos[i][1]), infos_to_make["size"]
            )], maior=True)

            local_pose = torch.cat([local_pose, local_pose[-1:]], dim=0)
        
        local_pose = local_pose[:-1]

    return local_pose

def make_pose(size, infos_json, tam_gauss=[48, 160]):
    infos_to_make = {
        "pose": [0, 1, 2, 3, 4, 5, 6, 7, 8],
        "face": [40, 41, 46, 47, 62, 66, 68, 69],
        "hand": [4, 8, 12, 16, 20],
        "size": size,
        "make": make_gauss_phoenix((size, size), tam_gauss=tam_gauss)
    }

    my_pose = torch.zeros(1, size, size)

    for k in infos_json['people'][0]:
        if k == 'person_id' or '3d' in k:
            continue

        if "pose" in k:
            m = make_gauss_matriz(infos_to_make, infos_json['people'][0][k], tipo="pose")
            my_pose = torch.cat([my_pose, m], dim=0)
        elif "hand" in k:
            m = make_gauss_matriz(infos_to_make, infos_json['people'][0][k], tipo="hand")
            my_pose = torch.cat([my_pose, m], dim=0)
        elif "face" in k:
            m = make_gauss_matriz(infos_to_make, infos_json['people'][0][k], tipo="face")
            my_pose = torch.cat([my_pose, m], dim=0)

    return my_pose[1:]

#32, 160

if __name__ == "__main__":
    from PIL import Image

    path_img = "/home/wellington/Downloads/780.png"

    pic = Image.open(path_img).convert('RGB')

    img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
    img = img.view(pic.size[1], pic.size[0], len(pic.getbands()))

    print(img.shape)

    imshow(img.numpy())


    '''
    import json, time

    path_json = "/home/wellington/Downloads/780_keypoints.json"

    with open(path_json, "r") as f:
        i = time.time()
        pose = make_pose(256, json.load(f), tam_gauss=[36, 176])
        print(pose.shape, time.time()-i)
        imshow(pose[1:4])

    img = make_gauss_phoenix(list_center=[
        (0, 0),
        (64,128),
        (100, 200)
    ], size_img=(256,256), size_gauss=160)

    print(img.shape)

    imshow(img)
    '''

    '''
    meus_gauss2 = torch.zeros(size_img, size_img, size_img, size_img)

    for i in range(size_img):
        for j in range(size_img):
            make_gauss(meus_gauss2[i,j], center=(i,j), std=(tam_gauss, tam_gauss))

        print(i)

    torch.save(meus_gauss2, name_file)

    g = torch.load(name_file)
    '''