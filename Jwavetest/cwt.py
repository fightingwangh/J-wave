import csv
import os
import cv2
import numpy as np
import pywt

# 指定的CSV文件名
csv_filename = 'data/data0.csv'    #0: J wave   1:non_J_wave

# 读取指定的CSV文件
with open(csv_filename, 'r', newline='\n') as f:
    reader = csv.reader(f)
    for idx, row in enumerate(reader):
        ecg_signal = np.array(row, dtype=np.float32)

        # 进行CWT变换
        wavelet = 'mexh'
        scales = np.arange(1, 128)

        coefficients, frequencies = pywt.cwt(ecg_signal, scales, wavelet)


        resized_coefficients = cv2.resize(np.abs(coefficients), (224, 224))


        scaled_coefficients = (resized_coefficients - np.min(resized_coefficients)) / (
                    np.max(resized_coefficients) - np.min(resized_coefficients)) * 255
        scaled_coefficients = scaled_coefficients.astype(np.uint8)


        output_folder = 'data/0'   #

        output_path = f'{output_folder}J_{idx + 1}.png'
        cv2.imwrite(output_path, scaled_coefficients)