import os
import sys
from urllib.request import urlretrieve
import urllib
from os import path
from logger import logger
import http


class FiveKDataset:
    """
    MIT-Adobe FiveK Dataset Automatically download pictures
    """

    def __int__(self):
        # Change current working path
        CURRENT_PATH = "E:\MyWorkspace\Image-Adaptive-3DLUT\dataset"  # Path of this file
        os.chdir(CURRENT_PATH)  # Change current path

    # Callback function of urlretrieve function, showing download progress
    def cbk(self, a, b, c):
        '''Callback
             @a: The number of downloaded data packages
             @b: The size of the data block
             @c: The size of the remote file
        '''
        per = 100.0 * a * b / c
        if per > 100:
            per = 100
        # Update progress in the terminal
        sys.stdout.write("progress: %.2f%%   \r" % (per))
        sys.stdout.flush()

    def download_single(self):
        CURRENT_PATH = "./dataset"  # Path of this file
        os.chdir(CURRENT_PATH)  # Change current path
        # A list of image names
        path = './FiveK_C/'
        img_lst = []
        img_lst_downloaded = os.listdir(path)
        # Read picture name list
        with open('filesAdobe.txt', 'r') as f:
            for line in f.readlines():
                img_lst.append(line.rstrip("\n"))  # Remove newlines

        with open('filesAdobeMIT.txt', 'r') as f:
            for line in f.readlines():
                img_lst.append(line.rstrip("\n"))  # Remove newlines
        # Download pictures according to the url of the file
        for i in img_lst:
            if img_lst_downloaded is not None and len(img_lst_downloaded) > 0 and i in img_lst_downloaded:
                print(i, "  already exists, skipping download")
                continue
            URL = 'https://data.csail.mit.edu/graphics/fivek/img/tiff16_c/' + i + '.tif'  # Download the image adjusted by C (the other four types of images can be downloaded as needed)
            print('Downloading ' + i + ':')
            # Store the acquired pictures in a local address
            urlretrieve(url=URL, filename=path + i + '.tif', reporthook=self.cbk)

    def download_parallel(self, total):
        """
        download in parallel model
        args:
          total:int, the total of tasks to execute download
        """
        import time
        import multiprocessing
        CURRENT_PATH = "./dataset"  # Path of this file
        os.chdir(CURRENT_PATH)  # Change current path
        # A list of image names
        path = './FiveK_C_p/'
        img_lst = []
        img_lst_downloaded = os.listdir(path)
        img_lst_downloaded = [i.rstrip('.tif') for i in img_lst_downloaded]
        # Read picture name list
        with open('filesAdobe.txt', 'r') as f:
            for line in f.readlines():
                img_lst.append(line.rstrip("\n"))  # Remove newlines

        with open('filesAdobeMIT.txt', 'r') as f:
            for line in f.readlines():
                img_lst.append(line.rstrip("\n"))  # Remove newlines
        # Download pictures according to the url of the file
        img_list_to_download = [i for i in img_lst if img_lst_downloaded is None or len(img_lst_downloaded) == 0 or (
                len(img_lst_downloaded) > 0 and i not in img_lst_downloaded)]
        time1 = time.time()
        _processes = []
        num = len(img_list_to_download)
        sub_num = num // total
        for index in range(total):
            sub_list = img_list_to_download[sub_num * index:sub_num * index + sub_num]
            _process = multiprocessing.Process(target=self.download_task, args=(sub_list, index, path))
            _process.start()
            _processes.append(_process)
        if num % total != 0:
            sub_list = img_list_to_download[sub_num * total:-1]
            _process = multiprocessing.Process(target=self.download_task, args=(sub_list, index, path))
            _process.start()
            _processes.append(_process)

        for _process in _processes:
            _process.join()
        time2 = time.time()
        print("time consume {} s".format(time2 - time1))

    def download_task(self, img_list, task_index, path='./FiveK_C/'):
        """
        downloading image.
        args:
          img_list: image name list
          task_index: subtask index
          path: the image will be saved here.
        """
        for i in img_list:
            URL = 'https://data.csail.mit.edu/graphics/fivek/img/tiff16_c/' + i + '.tif'  # Download the image adjusted by C (the other four types of images can be downloaded as needed)
            print('Task ', task_index, 'Downloading ' + i + ':')
            # Store the acquired pictures in a local address
            try:
                urlretrieve(url=URL, filename=path + i + '.tif', reporthook=self.cbk)
            except urllib.error.HTTPError as e:
                logger.error(e)
            except http.client.InvalidURL as e:
                logger.error(e)


if __name__ == '__main__':
    fiveK = FiveKDataset()
    fiveK.download_parallel(10)
    # url = 'https://connectpolyu-my.sharepoint.com/personal/16901447r_connect_polyu_hk/_layouts/15/download.aspx?UniqueId=39769750%2Df83e%2D400e%2Dbdb7%2D37d736aea352'
    # urlretrieve(url=url,filename='./test_one_driver.png',reporthook=fiveK.cbk)
