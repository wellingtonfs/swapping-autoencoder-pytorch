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
            #img = img / 2 + 0.5
            img = img.to("cpu")
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

def make_gauss_phoenix(list_center=[], size_img=(), size_gauss=10):
    assert len(size_img) == 2 and len(list_center) > 0 and size_gauss > 0

    num_to_par = 6 if size_gauss % 2 == 0 else 7
    size_gauss += num_to_par

    #gerar uma gaussiana
    gauss = torch.zeros(size_gauss, size_gauss)
    make_gauss(
        gauss,
        center=(size_gauss//2,size_gauss//2),
        amplitude=1,
        std=(size_gauss-num_to_par, size_gauss-num_to_par)
    )

    #criar imagem de saida
    img_full = torch.zeros(size_img[0]+size_gauss, size_img[1]+size_gauss)

    #rodar todos os pontos que deve ter uma gaussiana
    for ponto in list_center:
        x, y = ponto

        #como img_full é maior que a imagem de saida, temos que fixar o ponto 0,0 da imagem menor com
        #a compensação da imagem maior
        center = (x + size_gauss//2, y + size_gauss//2)

        xi, xf = center[0] - size_gauss//2, center[0] + size_gauss//2
        yi, yf = center[1] - size_gauss//2, center[1] + size_gauss//2

        img_full[xi:xf, yi:yf] += gauss

    return torch.transpose(
        img_full[size_gauss//2:-size_gauss//2, size_gauss//2:-size_gauss//2],
            1, 0).clamp_(0.0, 1.0)

def make_pose(size, infos_json, tam_gauss=[48, 160]):
    pose = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    face = [37, 38, 40, 41, 43, 44, 46, 47, 62, 66]
    hand = [4, 8, 12, 16, 20]

    def convert_to_tam_img(x, y):
        xx = int((x / 260.0) * size)
        yy = int((y / 260.0) * size)

        xx = xx if xx < size else size - 1
        yy = yy if yy < size else size - 1
        x = xx if xx >= 0 else 0
        y = yy if yy >= 0 else 0
        return x, y

    def make_gauss_matriz(pontos, all_pontos, maior=False, pose=False):
        local_pose = torch.zeros(1, size, size)

        ignorar_pontos = [1, 2, 3, 5, 6, 8] if pose else []

        all_pontos = torch.tensor(all_pontos).view(-1, 3)

        local_pose[0] = make_gauss_phoenix(list_center=[
            convert_to_tam_img(
                all_pontos[i][0],
                all_pontos[i][1]
            ) for i in range(all_pontos.shape[0]) if i in pontos and not i in ignorar_pontos
        ], size_img=(size, size), size_gauss=tam_gauss[1] if maior else tam_gauss[0])

        if pose:
            local_pose[0] += make_gauss_phoenix(list_center=[
                convert_to_tam_img(
                    all_pontos[i][0],
                    all_pontos[i][1]
                ) for i in range(all_pontos.shape[0]) if i in ignorar_pontos
            ], size_img=(size, size), size_gauss=tam_gauss[0])

        return local_pose

    my_pose = torch.zeros(1, size, size)

    for k in infos_json['people'][0]:
        if k == 'person_id' or '3d' in k:
            continue

        if "pose" in k:
            m = make_gauss_matriz(pose, infos_json['people'][0][k], maior=True, pose=True)
            my_pose = torch.cat([my_pose, m], dim=0)
        elif "hand" in k:
            m = make_gauss_matriz(hand, infos_json['people'][0][k])
            my_pose = torch.cat([my_pose, m], dim=0)
        elif "face" in k:
            m = make_gauss_matriz(face, infos_json['people'][0][k])
            my_pose = torch.cat([my_pose, m], dim=0)

    return my_pose[1:]

#32, 160

if __name__ == "__main__":
    import json

    path_json = "/home/wellington/Downloads/780_keypoints.json"

    with open(path_json, "r") as f:
        pose = make_pose(256, json.load(f))
        print(pose[:3].shape, pose[:3].max())
        imshow(pose[0])

    '''
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