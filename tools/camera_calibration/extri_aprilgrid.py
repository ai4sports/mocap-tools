import json

import numpy as np
import cv2
import sys
import os
from typing import List, Dict, Union
from utils import solvePnP, write_opencv_intri, write_opencv_extri, camera_project, CameraParam

try:
    import aprilgrid
    from aprilgrid import Detector
except ImportError:
    print('Please install the aprilgrid package')
    sys.exit(-1)


# generate apriltag coordinate
def apriltag_generator(tag_size: float,
                       space_ratio: float = 0.3,
                       row: int = 2,
                       col: int = 3,
                       ori_id: int = 0
                       ) -> dict:
    """
    generate apriltag coordinate, each tag has 4 points, each point has 3 coordinates, x, y, z in world coordinate,
    assume the tag is on the ground, z = 0, the origin is the center of the tag, the x axis is the horizontal axis
    Args:
        tag_size: the size of the tag in meter
        space_ratio: the ratio of the space between tags to the tag size, default 0.3, means the space between tags is 0.3 * tag_size
        row: number of tag rows, default 8
        col: number of tag cols, default 11
        ori_id: which tag is the origin, default 0, means the first tag is the origin

    Returns:
        tag_coord_dict: dict, the key is the tag id, the value is the tag coordinate,
        e.g: tag_coord_dict[0] = [[x0,y0,z0], [x1,y1,z1], [x2,y2,z2], [x3,y3,z3]]
    """
    tag_coord_dict = {}
    for i in range(row):
        for j in range(col):
            tag_id = i * col + j
            if tag_id == ori_id:
                x = 0
                y = 0
            else:
                x = (j - ori_id % col) * (tag_size + space_ratio * tag_size)
                y = (ori_id // col + i) * (tag_size + space_ratio * tag_size)
            tag_coord_dict[tag_id] = [[x, y, 0], [x + tag_size, y, 0], [x + tag_size, y + tag_size, 0],
                                      [x, y + tag_size, 0]]
    return tag_coord_dict


def get_tag_coordinate(tag_world_coord_dict: dict, apriltag_detection_list: list) -> (np.ndarray, np.ndarray):
    """
    get the tag coordinate in camera coordinate
    Args:
        tag_world_coord_dict: dict, the key is the tag id, the value is the tag coordinate in world coordinate,
        e.g: tag_coord_dict[0] = [[x0,y0,z0], [x1,y1,z1], [x2,y2,z2], [x3,y3,z3]]
        apriltag_detection_list: apriltag detection result

    Returns:
        world_coord: np.ndarray, the tag coordinate in world coordinate, shape is (n, 4, 3), n is the number of tags
        camera_coord: np.ndarray, the tag coordinate in camera coordinate, shape is (n, 4, 2), n is the number of tags
    """
    world_coord_list = []
    camera_coord_list = []
    tag_score = {}
    for tag in apriltag_detection_list:
        tag_id = tag.tag_id
        # skip 0, 10
        # if tag_id == 0 or tag_id == 10:
        #     continue
        corners = tag.corners
        if tag_id not in tag_world_coord_dict.keys():
            continue
        if tag_id in tag_score.keys():
            continue
        tag_score[tag_id] = 0
        world_coord = tag_world_coord_dict[tag_id]

        world_coord_list.append(world_coord)
        camera_coord_list.append(corners)
    world_coord = np.array(world_coord_list)
    camera_coord = np.array(camera_coord_list)
    return world_coord, camera_coord


def cal_extri(cam_points: np.ndarray, ground_points: np.ndarray, intri: CameraParam):
    # extri = {}
    k3d = ground_points.reshape(-1, 3)
    k2d = cam_points.reshape(-1, 2)
    K = np.array(intri["K"])
    dist = np.array(intri["dist"])

    err, rvec, tvec, kpts_repro = solvePnP(k3d, k2d, K, dist, flag=cv2.SOLVEPNP_ITERATIVE)

    # extri = {}
    camera_param_extri = CameraParam()
    camera_param_extri.update(intri)
    camera_param_extri['Rvec'] = rvec
    camera_param_extri['R'] = cv2.Rodrigues(rvec)[0]
    camera_param_extri['T'] = tvec
    center = - camera_param_extri['R'].T @ tvec
    print('center => {}, err = {:.3f}'.format(center.squeeze(), err))

    return camera_param_extri


def check_calibration_result(image: np.ndarray, camera: CameraParam, cube_coord: np.ndarray = None):
    """
    check the calibration result, draw the cube on the image
    Args:
        image: image
        camera_id: camera id
        intri: dict
        extri: dict
        cube_coord: np.ndarray, the coordinate of the cube in world coordinate, shape is (8, 3)

    Returns:
        image: image with cube
    """
    if cube_coord is None:
        # generate the cube coordinate, the cube is 1m * 1m * 1m
        cube_coord = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0],
                               [0, 1, 0], [0, 0, 1], [1, 0, 1],
                               [1, 1, 1], [0, 1, 1]])
    cube_coord = camera_project(cube_coord, camera)

    cube_coord = cube_coord.reshape(-1, 2).astype(np.int32)

    # draw the cube, first draw 8 points, then draw 12 lines
    for i in range(8):
        cv2.circle(image, tuple(cube_coord[i]), 3, (0, 0, 255), -1)
    lines = np.array([[0, 1], [1, 2], [2, 3], [3, 0],
                      [4, 5], [5, 6], [6, 7], [7, 4],
                      [0, 4], [1, 5], [2, 6], [3, 7]])
    for i in range(12):
        cv2.line(image, tuple(cube_coord[lines[i][0]]), tuple(cube_coord[lines[i][1]]), (0, 255, 0), 2)
    return image


def get_extri_from_apriltag_detection(apriltag_detection_list: list,
                                      intri: CameraParam,
                                      tag_size: float = 0.33,
                                      space_ratio: float = 0.3,
                                      row: int = 2,
                                      col: int = 3,
                                      ori_id: int = 0,
                                      ):
    """
    get extrinsic parameters from apriltag detection result
    Args:
        apriltag_detection_list: apriltag detection result
        intri: camera intrinsic parameters
        tag_size: the size of the tag in meter
        space_ratio: the ratio of the space between tags to the tag size, default 0.3, means the space between tags is 0.3 * tag_size
        row: number of tag rows, default 8
        col: number of tag cols, default 11
        ori_id: which tag is the origin, default 0, means the first tag is the origin

    Returns:
        extri: dict, the key is the camera id, the value is the extrinsic parameters
    """
    tag_world_coord_dict = apriltag_generator(tag_size, space_ratio, row, col, ori_id)
    world_coord, camera_coord = get_tag_coordinate(tag_world_coord_dict, apriltag_detection_list)
    if len(world_coord) == 0:
        return {}
    extri = cal_extri(camera_coord, world_coord, intri)
    return extri


def detect_apriltag(image: np.ndarray, tag_type:str='t36h11'):
    """
    detect apriltag with aprilgrid package
    Args:
        image: image
        tag_type: tag type, default 't36h11'
    """

    detector = Detector(tag_type, detect_max_size=4096,debug_level=3)
    tags = detector.detect(
        image,
    )
    return tags


def draw_tags(
        image: np.ndarray,
        tags: List[aprilgrid.detector.Detection],
):
    for tag in tags:
        tag_id = tag.tag_id
        corners = tag.corners.reshape(4, 2)
        # center = (int(center[0]), int(center[1]))
        center = (int(corners[0][0] + corners[2][0]) // 2,
                  int(corners[0][1] + corners[2][1]) // 2)
        corner_01 = (int(corners[0][0]), int(corners[0][1]))
        corner_02 = (int(corners[1][0]), int(corners[1][1]))
        corner_03 = (int(corners[2][0]), int(corners[2][1]))
        corner_04 = (int(corners[3][0]), int(corners[3][1]))

        # 中心
        cv2.circle(image, (center[0], center[1]), 5, (0, 0, 255), 2)

        # 各辺
        cv2.line(image, (corner_01[0], corner_01[1]),
                 (corner_02[0], corner_02[1]), (255, 0, 255), 2)
        cv2.line(image, (corner_02[0], corner_02[1]),
                 (corner_03[0], corner_03[1]), (255, 0, 0), 2)
        cv2.line(image, (corner_03[0], corner_03[1]),
                 (corner_04[0], corner_04[1]), (0, 255, 0), 2)
        cv2.line(image, (corner_04[0], corner_04[1]),
                 (corner_01[0], corner_01[1]), (0, 255, 0), 2)

        # タグファミリー、タグID
        # cv.putText(image,
        #            str(tag_family) + ':' + str(tag_id),
        #            (corner_01[0], corner_01[1] - 10), cv.FONT_HERSHEY_SIMPLEX,
        #            0.6, (0, 255, 0), 1, cv.LINE_AA)
        cv2.putText(image, str(tag_id), (center[0] - 10, center[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv2.LINE_AA)

    return image


def apriltag_calibration(image:np.ndarray, intri:CameraParam,  tag_size:float=0.09, space_ratio:float=0.3, row:int=4, col:int=4):
    """
    use apriltag to calibrate the extrinsic parameters
    Args:
        image: image
        camera_id: camera id
        intri: camera intrinsic parameters
        tag_size: the size of the tag in meter
        space_ratio: the ratio of the space between tags to the tag size, default 0.3, means the space between tags is 0.3 * tag_size
        row: number of tag rows, default 4
        col: number of tag cols, default 4
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    try:
        tags = detect_apriltag(gray)
    except Exception as e:
        print(e)
        tags = []
    if len(tags) == 0:
        return image, None
    draw_tags(image, tags)
    new_extri = get_extri_from_apriltag_detection(tags, intri, tag_size=tag_size, space_ratio=space_ratio, row=row, col=col)
    return image, new_extri



if __name__ == '__main__':
    image_dir = "E:\\DeblurHGS\\reorganized_images"
    intri_dir = "E:\\DeblurHGS\\camera_intris"
    output_camera_dir = "E:\\DeblurHGS\\camera_params"

    camera_image_dir = os.listdir(image_dir)
    for camera_id in camera_image_dir:
        image_path = os.path.join(image_dir, camera_id, f"{0:05d}.png")
        image = cv2.imread(image_path)
        intri_path = os.path.join(intri_dir, camera_id+".json")
        with open(intri_path, 'r') as f:
            intri = json.load(f)
            intri = CameraParam(**intri)
        image, extri = apriltag_calibration(image, intri)
        if extri is not None:
            cv2.imshow("image", image)
            cv2.waitKey(0)
            extri_path = os.path.join(output_camera_dir, camera_id+".json")
            with open(extri_path, 'w') as f:
                json.dump(extri.dump(), f, indent=2)
        else:
            print("Failed to calibrate the extrinsic parameters")
            cv2.imshow("image", image)
            cv2.waitKey(0)




