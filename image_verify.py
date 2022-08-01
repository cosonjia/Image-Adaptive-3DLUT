from PIL import Image
import sys
import os
from os import path


class ImageVerify:
    """
      图片完整性校验
    """

    def __init__(self, base_path):
        self.base_path = base_path

    @staticmethod
    def is_valid_image(image_path):
        try:
            Image.open(image_path).verify()
            return True
        except:
            return False

    def verify(self, remove_invalid=False):
        names = os.listdir(self.base_path)
        print(f"Found {len(names)} images")
        for name in names:
            if not self.is_valid_image(path.join(self.base_path, name)):
                print(name)
                if remove_invalid:
                    self.delete_valid_image(path.join(self.base_path, name))

    def delete_valid_image(self, image_path):
        if os.path.exists(image_path):
            if os.path.isfile(image_path):
                os.remove(image_path)
                print(f"Delete the {image_path} successfully!")
            else:
                print(f"Error: {image_path} is a directory!")
        else:
            print(f"{image_path} was not found or has been deleted!")


if __name__ == '__main__':
    base = '/home/pc10/MyWorkspace/Image-dataset/dataset/FiveK_C_p'
    img = ImageVerify(base)
    img.verify(remove_invalid=True)
