
import numpy as np

from collections import  OrderedDict
import xml.etree.ElementTree as ET




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

        tree = ET.parse(data_dir + '/cc_pose.xml')  # 解析读取xml函数
        camera_id = int(tree.getroot().find('Block').find('Photogroups').find('Photogroup').find('Name').text.split(' ')[-1])
        CameraModelType = tree.getroot().find('Block').find('Photogroups').find('Photogroup').find('CameraModelType').text
        CameraModelType = 'RADIAL' #只要k1,k2,详见https://www.jianshu.com/p/95cf3a63b6bb
        width = int(
            tree.getroot().find('Block').find('Photogroups').find('Photogroup').find('ImageDimensions').find('Width').text)
        height = int(
            tree.getroot().find('Block').find('Photogroups').find('Photogroup').find('ImageDimensions').find('Height').text)
        focal_length = float(
            tree.getroot().find('Block').find('Photogroups').find('Photogroup').find('FocalLengthPixels').text)
        cx = float(
            tree.getroot().find('Block').find('Photogroups').find('Photogroup').find('PrincipalPoint').find('x').text)
        cy = float(
            tree.getroot().find('Block').find('Photogroups').find('Photogroup').find('PrincipalPoint').find('y').text)

        params = {k: 0. for k in ['k1', 'k2', 'k3', 'p1', 'p2']}
        params['k1'] = float(
            tree.getroot().find('Block').find('Photogroups').find('Photogroup').find('Distortion').find('K1').text)
        params['k2'] = float(
            tree.getroot().find('Block').find('Photogroups').find('Photogroup').find('Distortion').find('K2').text)
        params['k3'] = float(
            tree.getroot().find('Block').find('Photogroups').find('Photogroup').find('Distortion').find('K3').text)
        params['p1'] = float(
            tree.getroot().find('Block').find('Photogroups').find('Photogroup').find('Distortion').find('P1').text)
        params['p2'] = float(
            tree.getroot().find('Block').find('Photogroups').find('Photogroup').find('Distortion').find('P2').text)

        # 1 SIMPLE_RADIAL 4032 3024 3062.7160769652855 2016 1512 0.025280661983932456
        #保存为cameras.txt
        data = {
            'camera_id': camera_id,
            'CameraModelType': CameraModelType,
            'width': width,
            'height': height,
            'focal_length': focal_length,
            'cx': cx,
            'cy': cy
        }

        # 打开文件并写入数据
        with open('../scripts/cameras.txt', 'w') as file:
            file.write("# Camera list with one line of data per camera:\n")
            file.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
            file.write("# Number of cameras: 1\n")
            file.write(
                f"{data['camera_id']} {data['CameraModelType']} {data['width']} {data['height']} {data['focal_length']} {data['cx']} {data['cy']} {params['k1']} {params['k2']}\n")
            # {params['k3']} {params['p1']} {params['p2']}


        with open('images.txt', 'w') as file:
            file.write("# Image list with two lines of data per image:\n")
            file.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
            file.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
            file.write("# Number of images: 73, mean observations per image: 0\n") #这里还得改
            for photo in tree.getroot().find('Block').find('Photogroups').find('Photogroup').findall('Photo'):
                pose_rotate_temp = []
                camera_id = int(
                    tree.getroot().find('Block').find('Photogroups').find('Photogroup').find('Name').text.split(' ')[
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
                image_id = int(photo.find('Id').text)
                q = ImageCC.FromR(np_pose_rotate)

                file.write(
                    f"{image_id} {q[0]} {q[1]} {q[2]} {q[3]} {np_pose_trans[0]} {np_pose_trans[1]} {np_pose_trans[2]} {camera_id} {img_name}\n")
                file.write('\n')


        with open('points3D.txt', 'w') as file:
            file.write("# 3D point list with one line of data per point:\n")
            file.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
            file.write("# Number of points: 0, mean track length: 0\n")



if __name__ == '__main__':
    CCposeLoader.process("/my_data/statue_0914")

