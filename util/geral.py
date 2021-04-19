import zipfile, os

def unzip(path_file, to_dir):
    if os.path.isdir(to_dir):
        return False
            
    with zipfile.ZipFile(path_file) as zf:
        zf.extractall(to_dir)

    return True