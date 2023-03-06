import random
def get_aug_image(path,diff_root_path):
    filename=path.split("/")[-1]
    classname=path.split("/")[-2]
    p = random.randint(0, 5)
    print(diff_root_path+"/"+classname+"/"+filename+"/img"+str(p)+".jpg")