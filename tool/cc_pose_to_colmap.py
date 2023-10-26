import os

import numpy as np

from collections import  OrderedDict
import xml.etree.ElementTree as ET

from internal import camera_utils


class ImageCC:
    def __init__(self, name_, camera_id_, q_, tvec_, rotation_=None):
        self.name = name_
        self.camera_id = camera_id_
        self.q = q_
        self.tvec = tvec_
        self.rotation = rotation_
        self.points2D = np.empty((0, 2), dtype=np.float64)
        self.point3D_ids = np.empty((0,), dtype=np.uint64)

    @staticmethod
    def FromR(R):
        trace = np.trace(R)

        if trace > 0:
            qw = 0.5 * np.sqrt(1. + trace)
            qx = (R[2, 1] - R[1, 2]) * 0.25 / qw
            qy = (R[0, 2] - R[2, 0]) * 0.25 / qw
            qz = (R[1, 0] - R[0, 1]) * 0.25 / qw
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2. * np.sqrt(1. + R[0, 0] - R[1, 1] - R[2, 2])
            qw = (R[2, 1] - R[1, 2]) / s
            qx = 0.25 * s
            qy = (R[0, 1] + R[1, 0]) / s
            qz = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2. * np.sqrt(1. + R[1, 1] - R[0, 0] - R[2, 2])
            qw = (R[0, 2] - R[2, 0]) / s
            qx = (R[0, 1] + R[1, 0]) / s
            qy = 0.25 * s
            qz = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2. * np.sqrt(1. + R[2, 2] - R[0, 0] - R[1, 1])
            qw = (R[1, 0] - R[0, 1]) / s
            qx = (R[0, 2] + R[2, 0]) / s
            qy = (R[1, 2] + R[2, 1]) / s
            qz = 0.25 * s

        return np.array((qw, qx, qy, qz))


class CCposeLoader():

    @staticmethod
    def process(data_dir):

        folder_path = '../my_data/Fly/sparse/0/'  # 指定新文件夹的路径

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print("文件夹创建成功！")
        else:
            print("文件夹已存在！")

        with open('../my_data/Fly/sparse/0/cameras.txt', 'a') as file:
            file.write("# Camera list with one line of data per camera:\n")
            file.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
            file.write("# Number of cameras: 4\n")

        with open('../my_data/Fly/sparse/0/images.txt', 'a') as file:
            file.write("# Image list with two lines of data per image:\n")
            file.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
            file.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
            file.write("# Number of images: 200, mean observations per image: 0\n") #这里还得改

        with open('../my_data/Fly/sparse/0/points3D.txt', 'a') as file:
            file.write("# 3D point list with one line of data per point:\n")
            file.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
            file.write("# Number of points: 0, mean track length: 0\n")


        tree = ET.parse(data_dir + '/cc_pose.xml')  # 解析读取xml函数

        first_intrinsic = None
        for photogroup in tree.getroot().find('Block').find('Photogroups').findall('Photogroup'):
            camera_id = int(photogroup.find('Name').text.split(' ')[-1])
            camtype = tree.getroot().find('Block').find('Photogroups').find('Photogroup').find('CameraModelType').text.lower()
            width = int(
                photogroup.find('ImageDimensions').find('Width').text)
            height = int(
                photogroup.find('ImageDimensions').find('Height').text)
            if photogroup.find('FocalLengthPixels') is not None:
                focal_length_pixels = float(
                    photogroup.find('FocalLengthPixels').text)
            else:
                # Focal length in pixels的计算方法
                # https://blog.csdn.net/qq_39861441/article/details/114980443
                focal_length = float(
                    photogroup.find('FocalLength').text)
                sensor_size = float(
                    photogroup.find('SensorSize').text)
                width = float(
                    photogroup.find('ImageDimensions').find('Width').text)
                focal_length_pixels = width * focal_length / sensor_size

            cx = float(
                photogroup.find('PrincipalPoint').find('x').text)
            cy = float(
                photogroup.find('PrincipalPoint').find('y').text)

            # 只要k1,k2,详见https://www.jianshu.com/p/95cf3a63b6bb
            if camtype == camera_utils.ProjectionType.PERSPECTIVE.value:
                params = {k: 0. for k in ['k1', 'k2', 'k3', 'p1', 'p2']}
                # params['k1'] = float(
                #     tree.getroot().find('Block').find('Photogroups').find('Photogroup').find('Distortion').find('K1').text)
                # params['k2'] = float(
                #     tree.getroot().find('Block').find('Photogroups').find('Photogroup').find('Distortion').find('K2').text)
                # params['k3'] = float(
                #     tree.getroot().find('Block').find('Photogroups').find('Photogroup').find('Distortion').find('K3').text)
                # params['p1'] = float(
                #     tree.getroot().find('Block').find('Photogroups').find('Photogroup').find('Distortion').find('P1').text)
                # params['p2'] = float(
                #     tree.getroot().find('Block').find('Photogroups').find('Photogroup').find('Distortion').find('P2').text)
            else:
                params = {k: 0. for k in ['k1','k2','p1', 'p2']}
                params['k1'] = float(
                    tree.getroot().find('Block').find('Photogroups').find('Photogroup').find('FisheyeDistortion').find(
                        'P1').text)
                params['k2'] = float(
                    tree.getroot().find('Block').find('Photogroups').find('Photogroup').find('FisheyeDistortion').find(
                        'P2').text)
                params['p1'] = float(
                    tree.getroot().find('Block').find('Photogroups').find('Photogroup').find('FisheyeDistortion').find(
                        'P3').text)
                params['p2'] = float(
                    tree.getroot().find('Block').find('Photogroups').find('Photogroup').find('FisheyeDistortion').find(
                        'P4').text)

            # 1 SIMPLE_RADIAL 4032 3024 3062.7160769652855 2016 1512 0.025280661983932456
            #保存为cameras.txt
            data = {
                'camera_id': camera_id,
                'camtype': camtype,
                'width': width,
                'height': height,
                'focal_length': focal_length_pixels,
                'cx': cx,
                'cy': cy
            }
            intrinsic = np.array([
                [focal_length_pixels, 0, cx],
                [0, focal_length_pixels, cy],
                [0, 0, 1]
            ])
            if camera_id == 1:
                first_intrinsic = intrinsic

            # 打开文件并写入数据
            with open('../my_data/Fly/sparse/0/cameras.txt', 'a') as file:
                file.write(
                    f"{data['camera_id']} {data['camtype']} {data['width']} {data['height']} {data['focal_length']} {data['cx']} {data['cy']} {params['k1']} {params['k2']} {params['p1']} {params['p2']}\n")
                #


            with open('../my_data/Fly/sparse/0/images.txt', 'a') as file:
                for photo in photogroup.findall('Photo'):
                    pose_rotate_temp = []
                    camera_id = int(
                        photogroup.find('Name').text.split(' ')[
                            -1])
                    img_name = photo.find('ImagePath').text.split('/')[-1]
                    trans_x = float(photo.find('Pose').find('Center').find('x').text)
                    trans_y = float(photo.find('Pose').find('Center').find('y').text)
                    trans_z = float(photo.find('Pose').find('Center').find('z').text)
                    pose_trans_temp = (trans_x, trans_y, trans_z)
                    pose_rotate_temp.append(float(photo.find('Pose').find('Rotation').find('M_00').text))
                    pose_rotate_temp.append(float(photo.find('Pose').find('Rotation').find('M_10').text))
                    pose_rotate_temp.append(float(photo.find('Pose').find('Rotation').find('M_20').text))
                    pose_rotate_temp.append(float(photo.find('Pose').find('Rotation').find('M_01').text))
                    pose_rotate_temp.append(float(photo.find('Pose').find('Rotation').find('M_11').text))
                    pose_rotate_temp.append(float(photo.find('Pose').find('Rotation').find('M_21').text))
                    pose_rotate_temp.append(float(photo.find('Pose').find('Rotation').find('M_02').text))
                    pose_rotate_temp.append(float(photo.find('Pose').find('Rotation').find('M_12').text))
                    pose_rotate_temp.append(float(photo.find('Pose').find('Rotation').find('M_22').text))
                    np_pose_rotate = np.array(pose_rotate_temp).reshape(3, 3).transpose()  # 3*3
                    np_pose_trans = np.array(pose_trans_temp)  # 3*1
                    np_pose_trans = -np.matmul(np_pose_rotate, np_pose_trans)  # 3*1
                    # 计算 first_intrinsic 的逆矩阵
                    first_intrinsic_inv = np.linalg.inv(first_intrinsic)

                    # 使用 @ 运算符进行矩阵乘法
                    np_pose_rotate = first_intrinsic_inv @ intrinsic @ np_pose_rotate
                    np_pose_trans = first_intrinsic_inv @ intrinsic @ np_pose_trans

                    image_id = int(photo.find('Id').text)
                    q = ImageCC.FromR(np_pose_rotate)

                    file.write(
                        f"{image_id} {q[0]} {q[1]} {q[2]} {q[3]} {np_pose_trans[0]} {np_pose_trans[1]} {np_pose_trans[2]} {camera_id} {img_name}\n")
                    file.write('\n')






if __name__ == '__main__':
    CCposeLoader.process("../my_data/Fly")

