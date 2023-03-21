import os
import numpy as np
from torch import tensor
import torch

def to_physic_camera(cam_K, focus_length, img_width, img_height):
    """
    Focus length can be any length.
    """
    fx, fy, cx, cy = cam_K[0], cam_K[4], cam_K[2], cam_K[5]

    physic_cam_param = {}
    physic_cam_param['sensor_sizeX'] = focus_length * img_width / fx;
    physic_cam_param['sensor_sizeY'] = focus_length * img_height / fy;

    physic_cam_param['sensor_shiftX'] = -(cx - img_width / 2.0) / img_width;
    physic_cam_param['sensor_shiftY'] = (cy - img_height / 2.0) / img_height;
    return physic_cam_param


def to_intrinsic_param(cam_K_matrix):
    cam_K = np.array(cam_K_matrix).reshape(-1)
    param = {}
    param['fx'] = cam_K[0]
    param['fy'] = cam_K[4]
    param['cx'] = cam_K[2]
    param['cy'] = cam_K[5]
    return param


def camera_2_canonical(points, cam_R_m2c, cam_T_m2c):
    """
    :param points: shape should be Nx3
    """
    return np.dot(cam_R_m2c.T ,(points.reshape(-1,3)-cam_T_m2c.reshape(1,3)).T).T # Nx3


def canonical_2_camera(points, cam_R_m2c, cam_T_m2c):
    return np.dot(cam_R_m2c, points.reshape(-1,3).T).T + cam_T_m2c.reshape(1,3)


def to_intrinsic_matrix(cam_K_param):
    cam_K_matrix = np.zeros(9)
    cam_K_matrix[0] = cam_K_param['fx']
    cam_K_matrix[4] = cam_K_param['fy']
    cam_K_matrix[2] = cam_K_param['cx']
    cam_K_matrix[5] = cam_K_param['cy']
    return cam_K_matrix.reshape(3,3)


def render_image(model, img_size, R, t, K, mode = 'rgb'):
    """
        img_size = (rgb.shape[1], rgb.shape[0])
    """
    from lynne_lib.pytless import renderer
    return renderer.render(model, img_size, K, R, t, mode=mode)


def add_rotation_shift(rot, x_angle, y_angle, z_angle, degrees = True):
    from scipy.spatial.transform import Rotation as R
    x_rot = R.from_euler('x', x_angle, degrees=degrees)
    rot = rot @ x_rot.as_matrix()
    y_rot = R.from_euler('y', y_angle, degrees=degrees)
    rot = rot @ y_rot.as_matrix()
    z_rot = R.from_euler('z', z_angle, degrees=degrees)
    rot = rot @ z_rot.as_matrix()
    return rot


def visulise_point_cloud(points, window_name = "Open3D", axis_size = 5):
    import open3d 
    #  draw open3d Coordinate system 
    axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_size, origin=[0, 0, 0])
    
    test_pcd = open3d.geometry.PointCloud()  #  Defining point clouds 
    test_pcd.points = open3d.utility.Vector3dVector(points)  #  Define the point cloud coordinate position 
    open3d.visualization.draw_geometries([test_pcd] + [axis_pcd], window_name=window_name)
    

def get_visible_point_cloud(point_cloud, cam_R_m2c, cam_T_m2c, visualise = False):
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcl_in_cam = canonical_2_camera(point_cloud, cam_R_m2c, cam_T_m2c)
    pcd.points = o3d.utility.Vector3dVector(pcl_in_cam)

    diameter = np.linalg.norm(
        np.asarray(pcd.get_max_bound()) - np.asarray(pcd.get_min_bound()))

    if np.sum(cam_T_m2c) == 0:
        camera = [0, 0, -diameter]
        radius = diameter * 100
    else:
        camera = [0,0,0] 
        radius = 5000000

    _, pt_map = pcd.hidden_point_removal(camera, radius)

    print("Visualize result")
    visible_pcd = pcd.select_by_index(pt_map)
    pcl_in_cano = camera_2_canonical(np.asarray(visible_pcd.points), np.eye(3), cam_T_m2c)
    if visualise:
        visulise_point_cloud(pcl_in_cano)
    return pcl_in_cano 


def show_point_cloud(points, axis_size = 10, window_name = 'Open3D', colors = None):
    import open3d as o3d
    
    axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_size, origin=[0, 0, 0])
    if isinstance(points, list):
        pcds = []
        for i in range(len(points)):
            point_cloud = points[i]
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(point_cloud)
            if colors is not None:
                color = np.tile(colors[i], point_cloud.shape[0]).reshape(-1,3)
                pcd.colors = o3d.utility.Vector3dVector(color)
            pcds.append(pcd)
    else:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        if colors is not None:
            color = np.tile(colors, points.shape[0]).reshape(-1,3)
            pcd.colors = o3d.utility.Vector3dVector(color)
        pcds = [pcd]
    o3d.visualization.draw_geometries(pcds + [axis_pcd], window_name = window_name)


def manul_project(kps, cam_K_matrix):
    cam_K = np.array(cam_K_matrix).reshape(-1)
    fx, fy, cx, cy = cam_K[0], cam_K[4], cam_K[2], cam_K[5]
    x = kps[0]*fx/kps[2] + cx
    y = kps[1]*fy/kps[2] + cy
    return (x, y)


def project_points(points_3d, cam_K_matrix, 
            cam_2_world_rot_vec = np.zeros(3), cam_2_world_t_vec = np.zeros(3), dist_coef = np.zeros(5)
        ):
    import cv2
    points_3d = points_3d.reshape(-1, 3)
    pixels = cv2.projectPoints(points_3d, cam_2_world_rot_vec, cam_2_world_t_vec, cam_K_matrix, dist_coef)[0]
    pixels = np.squeeze(pixels)
    return pixels.reshape(-1, 2)


def keep_bbox_in_image(bbx, image, enl=0):
    """
    :param bbx: x1, x2, y1, y2, in pixel coordinate, x in column, y in row
    """
    H, W = image.shape[:2] 
    x1, x2, y1, y2 = bbx
    x1 = int(max(x1, 0)) - enl
    x2 = int(min(x2, W)) + enl
    y1 = int(max(y1, 0)) - enl
    y2 = int(min(y2, H)) + enl
    return [x1,x2,y1,y2]


def depth_to_3D_coords(image, cam_K_matrix, bbox = None, 
                       sample_step = 1, to_point_cloud = False):
    """
    :param bbx: x1, x2, y1, y2, in pixel coordinate, x in column, y in row
    """
    x1, x2, y1, y2 = get_image_bbox(image, bbox = bbox)
    image = image[y1:y2, x1:x2]
    h, w = image.shape[:2]

    x_map = np.tile(np.array(range(x1, x2, sample_step)), (h,1))
    y_map = np.tile(np.array(range(y1, y2, sample_step)).reshape(h,1), (1,w))

    cam_params = to_intrinsic_param(cam_K_matrix)
    real_x = (x_map - cam_params['cx'])*image/cam_params['fx']
    real_y = (y_map - cam_params['cy'])*image/cam_params['fy']
    new_image = np.stack((real_x, real_y, image), axis=2)
    if to_point_cloud:
        new_image = new_image.reshape(-1,3)
        new_image = new_image[np.where(new_image[:, 2] > 0.0)]
    return new_image



def get_image_bbox(image, bbox = None):
    if bbox is not None:
        x1, x2, y1, y2 = keep_bbox_in_image(bbox, image)
    else:
        h, w = image.shape[:2]
        x1, x2, y1, y2 = 0, w, 0, h
    return [x1, x2, y1, y2]


def to_open3d_point_cloud(points, calc_normal = False, visualise = False):
    """
    :param points: a numpy array with shape (N, 3) 
    :param calc_normal: will calculate and add the normal into open3d PointCloud instance.
    :param visualise: show the resutling mesh if visualise is True.
    """
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if calc_normal:
        normals = get_pcl_normals(points)
        pcd.normals = o3d.utility.Vector3dVector(normals)
    if visualise:
        o3d.visualization.draw_geometries([pcd])
    return pcd


def get_pcl_normals(points):
    """
    :param points: a numpy array with shape (N, 3) 
    """
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.normals = o3d.utility.Vector3dVector(np.zeros(
    (1, 3)))
    pcd.estimate_normals()
    pcd.orient_normals_consistent_tangent_plane(100)
    return np.asarray(pcd.normals)


def point_cloud_to_poisson_mesh(open3d_pcd, depth = 9, 
                                density_threshold = None, 
                                visualise = False,
                                axis_size = 5):
    """
    :param open3d_pcd: point cloud instance of open3d.
    :param visualise: show the resutling mesh if visualise is True.
    """
    import open3d as o3d
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                                                            open3d_pcd, depth=depth)
    if density_threshold is not None:
        vertices_to_remove = densities < np.quantile(densities, density_threshold)
        mesh.remove_vertices_by_mask(vertices_to_remove)
    if visualise:
        axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_size, origin=[0, 0, 0])
        o3d.visualization.draw_geometries([open3d_pcd, mesh, axis_pcd], mesh_show_back_face=True)
    return mesh


def point_cloud_to_alpha_mesh(open3d_pcd, alpha=None, 
                              alpha_log_range = (0.5, 0.01), 
                              test_num = 4, 
                              visualise = False,
                              axis_size = 5):
    """
    :param open3d_pcd: point cloud instance of open3d.
    :param visualise: show the resutling mesh if visualise is True.
    """
    import open3d as o3d
    tetra_mesh, pt_map = o3d.geometry.TetraMesh.create_from_point_cloud(open3d_pcd)
    if alpha == None:
        alpha_list = np.logspace(np.log10(alpha_log_range[0]), 
                                 np.log10(alpha_log_range[1]), 
                                 num=test_num)
    else:
        alpha_list = [alpha]

    for alpha in alpha_list:
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
                open3d_pcd, alpha, tetra_mesh, pt_map)
        mesh.compute_vertex_normals()
        if visualise:
            axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_size, origin=[0, 0, 0])
            o3d.visualization.draw_geometries([open3d_pcd, mesh, axis_pcd], 
                                              mesh_show_back_face=True)
    return mesh


def point_cloud_to_ball_pivoting_mesh(open3d_pcd, rad_range_list, 
                                      visualise = False, axis_size = 5):
    """
    :param open3d_pcd: point cloud instance of open3d.
    :param rad_range_list: a list contains the ranges of ball radius for mesh reconstruction.
                           Example:
                           rad_range_list = [
                               {'low': 0.0001,  'high': 0.001, 'step': 0.00002},
                               {'low': 0.001,   'high': 0.01,  'step': 0.0002 },
                               {'low': 0.01,    'high': 0.06,  'step': 0.002  }
                           ]
    :param visualise: show the resutling mesh if visualise is True.
    """
    import open3d as o3d

    radii = []
    for rad_range in rad_range_list:
        radii += np.arange(rad_range['low'], 
                           rad_range['high'], 
                           step = rad_range['step']).tolist()

    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                                                open3d_pcd, 
                                                o3d.utility.DoubleVector(radii))
    if visualise:
        axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_size, origin=[0, 0, 0])
        o3d.visualization.draw_geometries([open3d_pcd, mesh, axis_pcd], mesh_show_back_face=True)
    return mesh


def obs_rot_2_true_rot(obs_R, cam_T_m2c, return_rot_shift=False):
    rot_shift = get_visual_rot_shift(cam_T_m2c)
    rot = np.matmul(rot_shift.T, obs_R)
    if return_rot_shift:
        return rot, rot_shift
    return rot 


def true_rot_2_observe_rot(true_R, cam_T_m2c, return_rot_shift=False):
    rot_shift = get_visual_rot_shift(cam_T_m2c)
    rot = np.matmul(rot_shift, true_R)
    if return_rot_shift:
        return rot, rot_shift
    return rot 


def get_visual_rot_shift(cam_T_m2c): 
    from tools import transform
    z_axis = np.array([0,0,1])
    eps = 1e-7
    cam_T_m2c = cam_T_m2c.reshape(3,)
    normed_T = cam_T_m2c/np.linalg.norm(cam_T_m2c)
    if np.linalg.norm(normed_T - z_axis) <= eps:
        return np.eye(3)
    cosin_theta = np.matmul(normed_T, z_axis)
    angle = np.arccos(cosin_theta)
    axis = np.cross(normed_T, z_axis)
    rotation = transform.rotation_matrix(angle, axis)[:3,:3]
    return rotation


def back_project_pixel(pixel, depth, cam_K_matrix):
    cam_params = to_intrinsic_param(cam_K_matrix)
    px, py = pixel
    real_x = (px - cam_params['cx'])*depth/cam_params['fx']
    real_y = (py - cam_params['cy'])*depth/cam_params['fy']
    return np.array([real_x, real_y, depth])


def correct_pcl_rot_shift_use_pixel(pcl, pixel, cam_K_matrix, to_center = False):
    depth = 0.5*(np.max(pcl[:,2])+np.min(pcl[:,2])) # depth range dose not influence rotaiton shift
    rot_shift_point = back_project_pixel(pixel, depth, cam_K_matrix)
    return correct_pcl_rot_shift_use_T(pcl, rot_shift_point, to_center = to_center)


def correct_pcl_rot_shift_use_T(pcl, cam_T_m2c, to_center = False):
    rot_shift = get_visual_rot_shift(cam_T_m2c)
    pcl_new = camera_2_canonical(pcl, rot_shift.T, cam_T_m2c)
    if not to_center:
        pcl_new = canonical_2_camera(pcl_new, np.eye(3), cam_T_m2c)
    return pcl_new

    
def combine_image(rgb, rgb_top):
    vis_rgb = 0.7 * rgb_top + 0.3 * rgb
    vis_rgb[vis_rgb > 255] = 255
    return vis_rgb.astype(np.uint8)


def plot_sphere_heatmap(point_list, data_type = 'numpy'):
    import matplotlib.pyplot as plt
    from matplotlib import cm
    
    def near( p, pntList, d0 ):
        cnt=0
        threshold = 1 - d0
        for pj in pntList:
            dist = p.reshape(1,3) @ pj.reshape(3,1)
            if dist > threshold:
                cnt += 1 - (1-dist)/d0 
        return cnt
    
    def near_torch(p, pntList, d0):
        import torch
        threshold = 1 - d0
        dist = torch.matmul(p.reshape(1, 3).float(), pntList.reshape(-1, 3, 1).float())
        mask = dist>threshold
        count = (1 - (1-dist[mask])/d0).sum()
        return count.cpu().numpy()
        

    fig = plt.figure()
    ax = fig.add_subplot( 1, 1, 1, projection='3d')

    u = np.linspace( 0, 2 * np.pi, 120)
    v = np.linspace( 0, np.pi, 60 )

    # create the sphere surface
    XX = 1 * np.outer( np.cos( u ), np.sin( v ) )
    YY = 1 * np.outer( np.sin( u ), np.sin( v ) )
    ZZ = 1 * np.outer( np.ones( np.size( u ) ), np.cos( v ) )

    WW = XX.copy()
    for i in range( len( XX ) ):
        for j in range( len( XX[0] ) ):
            x = XX[ i, j ]
            y = YY[ i, j ]
            z = ZZ[ i, j ]
            query_vec = np.array([x, y, z ])
            if data_type == 'tensor':
                import torch
                WW[ i, j ] = near_torch(torch.from_numpy(query_vec).cuda(), point_list, 0.005)
            else:
                WW[ i, j ] = near(np.array(query_vec), point_list, 0.005) # about 5 degree
    print('************', np.amax(WW))
    WW = WW / (np.amax( WW ))
    myheatmap = WW

    ax.plot_surface( XX, YY,  ZZ, cstride=1, rstride=1, facecolors=cm.jet( myheatmap ) )
    plt.show() 


def rot_matrix_to_view_points(rot_mats):
    import torch
    """ Turn batched rotation matrices to 3D view point vectors.

    Args:
        rot_mats (torch.tensor, dtype=float): last two dimentions should be 3x3.

    Returns:
        viewpoints (torch.tensor, dtype=float): last dimention is 3 
    """
    assert len(rot_mats.size()) == 3 
    assert rot_mats.size()[-2:] == (3,3) 
    data_num = rot_mats.size(0)
    unit_vec = torch.tensor([0,0,1], dtype=float)
    viewpoints = torch.matmul(rot_mats, unit_vec.reshape(3,1))
    
    return viewpoints.reshape(data_num,3)

def crop_pointcloud(points, zmin, zmax, return_mask = False):
    depths = points[:, 2]
    pts_mask = (depths < zmax) & (depths > zmin)
    res_pts = points[pts_mask]
    
    if isinstance(points, np.ndarray):
        res_pts = res_pts.copy()

    if return_mask:
        return res_pts, pts_mask
    return res_pts 

def crop_pointcloud_with_radius(points, center_point, radius = 400, return_mask = False):

    def get_depth_range(center_point, depth_radius):
        center_depth = center_point[2]
        zmin = max(0, center_depth - depth_radius)
        zmax = center_depth + depth_radius
        return zmin, zmax

    zmin, zmax = get_depth_range(center_point, radius)
    return crop_pointcloud(points, zmin, zmax, return_mask = return_mask)

def add_pose_noise(points, ori_R, ori_T, x_range = 10, y_range=10, z_range=20):
    """ Add rotation and translation noise to a pointcloud based on the visual relationship. The rotation is changed with translation accordingly.

    Args:
        points (numpy.array): the pointcloud of an object. Shape: (N, 3), where N is the point number.
        ori_R (numpy.array): the ground truth rotation of the object in the pointcloud. Shape: (3,3)
        ori_T (numpy.array): the ground truth rotation of the object in the pointcloud. Shape: (3,).
        x_range (int): the noise_range of x in translation. 
        y_range (int): the noise_range of y in translation. 
        z_range (int): the noise_range of z in translation. 

    Returns:
        _type_: _description_
    """
    obs_rot, ori_rot_shift = true_rot_2_observe_rot(ori_R, ori_T, return_rot_shift=True)
    obs_points = np.matmul(ori_rot_shift, (points-ori_T.reshape(1,3)).T).T
    
    trans_noise = np.array([np.random.uniform(-x_range, x_range),
                            np.random.uniform(-y_range, y_range),
                            np.random.uniform(-z_range, z_range) ])
    new_T = ori_T + trans_noise
    new_R, new_rot_shift = obs_rot_2_true_rot(obs_rot, new_T, return_rot_shift=True)

    new_points = np.matmul(new_rot_shift.T, obs_points.T).T + new_T.reshape(1,3)

    rot_noise = np.matmul(new_rot_shift.T, ori_rot_shift)
    return new_points, new_R, new_T, rot_noise