import numpy as np
import cv2
import os
from typing import List, Dict, Union, TypedDict


# class CameraParam(TypedDict):
#     K: np.ndarray
#     invK: np.ndarray
#     H: int
#     W: int
#     RT: np.ndarray
#     R: np.ndarray
#     Rvec: np.ndarray
#     T: np.ndarray
#     P: np.ndarray
#     dist: np.ndarray
class CameraParam:
    def __init__(self,
                 K: Union[List[List[float]], np.ndarray] = None,
                 invK: Union[List[List[float]], np.ndarray] = None,
                 H: int = -1,
                 W: int = -1,
                 RT: Union[List[List[float]], np.ndarray] = None,
                 R: Union[List[List[float]], np.ndarray] = None,
                 Rvec: Union[List[float], np.ndarray] = None,
                 T: Union[List[List[float]], List[float], np.ndarray] = None,
                 P: Union[List[List[float]], np.ndarray] = None,
                 dist: Union[List[List[float]], List[float], np.ndarray] = None,
                 ):
        self.__setitem__('K', K)
        self.__setitem__('invK', invK)
        self.__setitem__('H', H)
        self.__setitem__('W', W)
        self.__setitem__('RT', RT)
        self.__setitem__('R', R)
        self.__setitem__('Rvec', Rvec)
        self.__setitem__('T', T)
        self.__setitem__('P', P)
        self.__setitem__('dist', dist)

    def __setitem__(self, key, value):
        if isinstance(value, List):
            value = np.array(value)
        setattr(self, key, value)

    def __getitem__(self, item):
        return getattr(self, item)

    def __str__(self):
        return str(self.__dict__)


    def update(self, kwargs):
        if isinstance(kwargs, CameraParam):
            kwargs = kwargs.__dict__
        for key, value in kwargs.items():
            self.__setitem__(key, value)

    def dump(self):
        d = {}
        for key in self.__dict__.keys():
            if isinstance(self.__dict__[key], np.ndarray):
                d[key] = self.__dict__[key].tolist()
            else:
                d[key] = self.__dict__[key]
        return d


MultiCameraParam = Dict[str, CameraParam]


def solvePnP(k3d, k2d, K, dist, flag, tryextri=False):
    k2d = np.ascontiguousarray(k2d[:, :2])
    # try different initial values:
    if tryextri:
        def closure(rvec, tvec):
            ret, rvec, tvec = cv2.solvePnP(k3d, k2d, K, dist, rvec, tvec, True, flags=flag)
            points2d_repro, xxx = cv2.projectPoints(k3d, rvec, tvec, K, dist)
            kpts_repro = points2d_repro.squeeze()
            err = np.linalg.norm(points2d_repro.squeeze() - k2d, axis=1).mean()
            return err, rvec, tvec, kpts_repro

        # create a series of extrinsic parameters looking at the origin
        height_guess = 2.1
        radius_guess = 7.
        infos = []
        for theta in np.linspace(0, 2 * np.pi, 180):
            st = np.sin(theta)
            ct = np.cos(theta)
            center = np.array([radius_guess * ct, radius_guess * st, height_guess]).reshape(3, 1)
            R = np.array([
                [-st, ct, 0],
                [0, 0, -1],
                [-ct, -st, 0]
            ])
            tvec = - R @ center
            rvec = cv2.Rodrigues(R)[0]
            err, rvec, tvec, kpts_repro = closure(rvec, tvec)
            infos.append({
                'err': err,
                'repro': kpts_repro,
                'rvec': rvec,
                'tvec': tvec
            })
        infos.sort(key=lambda x: x['err'])
        err, rvec, tvec, kpts_repro = infos[0]['err'], infos[0]['rvec'], infos[0]['tvec'], infos[0]['repro']
    else:
        ret, rvec, tvec = cv2.solvePnP(k3d, k2d, K, dist, flags=flag)
        points2d_repro, xxx = cv2.projectPoints(k3d, rvec, tvec, K, dist)
        kpts_repro = points2d_repro.squeeze()
        err = np.linalg.norm(points2d_repro.squeeze() - k2d, axis=1).mean()
    # print(err)
    return err, rvec, tvec, kpts_repro


def write_opencv_intri(intri_name, cameras):
    if not os.path.exists(os.path.dirname(intri_name)):
        os.makedirs(os.path.dirname(intri_name))
    intri = cv2.FileStorage(intri_name, True)
    results = {}
    camnames = list(cameras.keys())
    intri.write('names', camnames, 'list')
    for key_, val in cameras.items():
        key = key_.split('.')[0]
        K, dist = val['K'], val['dist']
        assert K.shape == (3, 3), K.shape
        assert dist.shape == (1, 5) or dist.shape == (5, 1) or dist.shape == (1, 4) or dist.shape == (4, 1), dist.shape
        intri.write('K_{}'.format(key), K)
        intri.write('dist_{}'.format(key), dist.flatten()[None])


def write_opencv_extri(extri_name, cameras):
    if not os.path.exists(os.path.dirname(extri_name)):
        os.makedirs(os.path.dirname(extri_name))
    extri = cv2.FileStorage(extri_name, True)
    results = {}
    camnames = list(cameras.keys())
    extri.write('names', camnames, 'list')
    for key_, val in cameras.items():
        key = key_.split('.')[0]
        extri.write('R_{}'.format(key), val['Rvec'])
        extri.write('Rot_{}'.format(key), val['R'])
        extri.write('T_{}'.format(key), val['T'])
    return 0


def camera_project(points: np.ndarray, camera: dict) -> np.ndarray:
    """
    project 3d points to 2d points
    Args:
        points: np.ndarray, [n_points, 3]
        camera: {"R":np.ndarray [3,3], "T":np.ndarray [3,1], "K":np.ndarray [3,3], "dist":np.ndarray [1,5]}

    Returns:
        points2d: np.ndarray, [n_points, 2]
    """
    r_mat = camera["R"].astype("float64")
    t_vec = camera["T"].astype("float64")
    K = camera["K"].astype("float64")
    dist = camera["dist"].astype("float64")
    points2d, _ = cv2.projectPoints(points.astype("float64"), r_mat, t_vec, K, dist)

    return points2d.reshape(-1, 2).astype("float64")
