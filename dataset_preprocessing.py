import cv2
import os
import tqdm
import matplotlib.pyplot as plt
from multiprocessing import Pool


def disp_img(img):
    plt.figure(figsize=(15,15))
    plt.imshow(img)
    plt.show()


def split_array_in_sub(array, n_split, brute_mode=False):
    if brute_mode:
        result = []
        for i in range(0, 18):
            mod = 5000
            arr = array[mod + ((i - 1) * mod):i * mod + mod]
            result.append(arr)
        return result

    avg = len(array) / float(n_split)
    result = []
    last = 0.0

    while last < len(array):
        result.append(array[int(last):int(last + avg)])
        last += avg

    result.sort()

    return result


def extract_rgb_from_greyscales(data_path, output_path, parrallelize=False, n_jobs=None, display_channels=False, display_logs=False):
    if parrallelize and n_jobs == None:
        print('Error, specify n_jobs to parrallelize')
        return

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    file_list = os.listdir(data_path)
    file_list.sort()

    if verify_hpa_greyscale_file_list(file_list) == False:
        print('Error, something is wrong with the files')
        return

    if parrallelize:
        qty = len(file_list)
        chunks = split_array_in_sub(file_list, n_jobs, brute_mode=True)
        pool = Pool()
        results = []
        for chunk in chunks:
            #print('verify : ', verify_hpa_greyscale_file_list(chunk), len(chunk))
            #r = [pool.apply_async(_extract_rgb_from_greyscales, (chunk, data_path, output_path, display_channels, display_logs))]
            results.append(pool.apply_async(_extract_rgb_from_greyscales, [chunk, data_path, output_path, display_channels, display_logs]))
        for r in results:
            r.get(timeout=60*60)
    else:
        _extract_rgb_from_greyscales(file_list, data_path, output_path, display_channels, display_logs)


def verify_hpa_greyscale_file_list(file_list, print_mode=False):
    if len(file_list) % 4 != 0:
        print('ERROR : not \%4')
        return
    for i in range(0, len(file_list), 4):
        if file_list[i].split('_')[0] == file_list[i + 1].split('_')[0] == file_list[i + 2].split('_')[0] == \
                file_list[i + 3].split('_')[0]:
            continue
        else:
            if (print_mode):
                print('ERROR : names do not match')
            return False
    return True


def _extract_rgb_from_greyscales(file_list, data_path, output_path, display_channels=False, display_logs=False):

    range_upper_bound = int(len(file_list) / 4)

    for i in tqdm.tqdm(range(0, range_upper_bound)):
        yellow = file_list.pop()
        red = file_list.pop()
        green = file_list.pop()
        blue = file_list.pop()

        y = cv2.imread(data_path + '/' + yellow, cv2.IMREAD_GRAYSCALE)
        r = cv2.imread(data_path + '/' + red, cv2.IMREAD_GRAYSCALE)
        g = cv2.imread(data_path + '/' + green, cv2.IMREAD_GRAYSCALE)
        b = cv2.imread(data_path + '/' + blue, cv2.IMREAD_GRAYSCALE)

        if yellow.split('_')[0] == red.split('_')[0] == green.split('_')[0] == blue.split('_')[0]:
            file_name = yellow.split('_')[0]
            img = cv2.merge([b, g, r])
            cv2.imwrite(output_path + '/' + file_name + '.png', img)
            if display_channels:
                disp_img(y)
                disp_img(r)
                disp_img(g)
                disp_img(b)
                disp_img(img)


if __name__ == '__main__':
    data_path = 'F:/hpa-single-cell-image-classification/train/'
    output_path = 'E:/hpa_train_rgb/'
    file_list = os.listdir(data_path)
    file_list.sort()

    extract_rgb_from_greyscales(data_path, output_path, display_channels=False, parrallelize=True, n_jobs=18)
