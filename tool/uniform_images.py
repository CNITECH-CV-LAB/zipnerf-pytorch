from math import inf

import cv2
import os
import numpy as np

cameras = []


def process(data_dir):
    with open(data_dir + '/sparse/1/cameras.txt', 'r') as file:
        lines = file.readlines()
        for line in lines[3:7]:
            camera_data = line.strip().split()
            cameras.append({
                'CAMERA_ID': int(camera_data[0])-1,
                'MODEL': camera_data[1],
                'WIDTH': int(float(camera_data[2])),
                'HEIGHT': int(float(camera_data[3])),
                'FOCAL_LENGTH': float(camera_data[4]),
                'CX': float(camera_data[5]),
                'CY': float(camera_data[6])
            })

    # 计算新相机的属性值
    new_camera = {
        'CAMERA_ID': 5,
        'MODEL': 'perspective',
        'WIDTH': min(camera['WIDTH'] for camera in cameras),
        'HEIGHT': min(camera['HEIGHT'] for camera in cameras),
        'FOCAL_LENGTH': max(camera['FOCAL_LENGTH'] for camera in cameras),
        'CX': min(camera['WIDTH'] for camera in cameras)/2,
        'CY': min(camera['HEIGHT'] for camera in cameras)/2
    }
    cameras.append(new_camera)

    # cam1-4 图像文件夹路径
    image_folder = data_dir + '/images'
    # 新文件夹路径
    output_folder = image_folder + '/uniformed'
    os.makedirs(output_folder, exist_ok=True)
    min_width = inf
    min_height = inf
    # 获取最小宽度和最小高度
    for camera in cameras:

        if camera['CAMERA_ID']< 4:
            cx = camera['CX']
            cy = camera['CY']

            # 平移图像
            tx = int(cx - camera['WIDTH'] / 2)
            ty = int(cy - camera['HEIGHT'] / 2)

            if tx >= 0:
                crop_width = camera['WIDTH'] - tx
            else:
                crop_width = camera['WIDTH'] + tx
            if ty >= 0:
                crop_height = camera['HEIGHT'] - ty
            else:
                crop_height = camera['HEIGHT'] + ty
            scale = cameras[4]['FOCAL_LENGTH'] / camera['FOCAL_LENGTH']
            crop_width = int(crop_width * scale)
            crop_height = int(crop_height * scale)
            min_width = crop_width if crop_width < min_width else min_width
            min_height = crop_height if crop_height < min_height else min_height

    cameras[4]['WIDTH'] = min_width
    cameras[4]['HEIGHT'] = min_height
    cameras[4]['CX'] = int(min_width/2)
    cameras[4]['CY'] = int(min_height/2)
    print(cameras[4]['WIDTH'])
    print(cameras[4]['HEIGHT'])

    with open(data_dir + '/sparse/0/cameras1.txt', 'a') as file:
        file.write("# Camera list with one line of data per camera:\n")
        file.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        file.write("# Number of cameras: 4\n")
        file.write(
            f"{cameras[4]['CAMERA_ID']} {cameras[4]['MODEL']} {cameras[4]['WIDTH']} {cameras[4]['HEIGHT']} {cameras[4]['FOCAL_LENGTH']} {cameras[4]['CX']} {cameras[4]['CY']} 0 0 0 0\n")
        #

    # 处理每张图像
    for filename in os.listdir(image_folder):
        if filename.endswith(".jpg"):
            image_path = os.path.join(image_folder, filename)
            image = cv2.imread(image_path)

            # 获取图像对应的相机ID
            camera_id = int(filename.split("-")[1].split(".")[0][3:])

            # 获取相机ID对应的内参
            if camera_id < 4:
                camera = cameras[camera_id]
                focal_length = camera['FOCAL_LENGTH']
                cx = camera['CX']
                cy = camera['CY']

                # 平移图像
                tx = int(cx - camera['WIDTH'] / 2)
                ty = int(cy - camera['HEIGHT'] / 2)
                M = np.float32([[1, 0, tx], [0, 1, ty]])
                translated_image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

                # 裁剪图像
                cropped_image = translated_image[
                    int(max(0, ty)):int(ty + camera['HEIGHT']),
                    int(max(0, tx)):int(tx + camera['WIDTH'])
                ]

                cropped_height, cropped_width, _ = cropped_image.shape

                # 缩放图像
                scale = cameras[4]['FOCAL_LENGTH'] / camera['FOCAL_LENGTH']
                scale_width = cropped_width * scale
                scale_height = cropped_height * scale
                resized_image = cv2.resize(cropped_image, (int(scale_width), int(scale_height)))

                # Center crop the image to the specified dimensions
                top = int((scale_height - min_height) / 2)
                bottom = int((scale_height + min_height) / 2)
                left = int((scale_width - min_width) / 2)
                right = int((scale_width + min_width) / 2)
                cropped_resized_image = resized_image[top:bottom, left:right]

                # 保存裁剪后的图像到输出文件夹
                output_path = os.path.join(output_folder, filename)
                cv2.imwrite(output_path, cropped_resized_image)


if __name__ == '__main__':
    process("../my_data/Fly")