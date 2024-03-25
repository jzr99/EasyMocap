import os
import shutil
import numpy as np
import cv2
import open3d as o3d
from os.path import join
from os.path import join
from glob import glob
from tqdm import tqdm
from easymocap.mytools.file_utils import read_json, write_keypoints3d, write_vertices, write_smpl
import shutil


def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()
    out = cv2.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]
    K = K/K[2,2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3,3] = (t[:3] / t[3])[:,0]
    return intrinsics, pose

# Input and output path
database = '/media/ubuntu/hdd/Hi4D'
outdatabase = '/media/ubuntu/hdd/easymocap/EasyMocap/data/V2As_easymocap'
# v2as_raw_root = '/media/ubuntu/hdd/RGB-PINA/preprocessing/raw_data'
v2as_data_root = '/media/ubuntu/hdd/RGB-PINA/data'

# After convering, use
#     python3 apps/calibration/vis_camera_by_open3d.py ${data} --pcd ${data}/mesh-test.obj
# for visualization

cam_path = '/media/ubuntu/hdd/Hi4D/pair19/piggyback19/cameras/rgb_cameras.npz'
current_view = 4
seqlist = [
    'Hi4D_pair19_piggyback19_static'
    # 'pair17_dance17_vitpose_28',
    # 'pair19_piggyback19_vitpose_4',
    # 'pair16_jump16_vitpose_4'
    # 'pair16/jump16'
    # 'pair10/dance10',
    # 'pair32/pose32',
    # 'pair09/hug09',
    # 'pair00/fight00',
    # 'pair00/hug00',
    # 'pair12/fight12',
    # 'pair12/hug12',
    # 'pair14/dance14',
    # 'pair28/dance28',
    # 'pair32/pose32',
    # 'pair37/pose37',
]


from easymocap.bodymodel.smpl import SMPLModel
body_model = SMPLModel(
    model_path='data/bodymodels/SMPL_python_v.1.1.0/smpl/models/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl',
    device = 'cuda',
    regressor_path = 'data/smplx/J_regressor_body25.npy',
    NUM_SHAPES = 10,
    use_pose_blending =True
)

for seq in seqlist:
    # root_raw = join(v2as_raw_root, seq)
    root_data = join(v2as_data_root, seq)
    outroot = join(outdatabase, seq.replace('/', '_'))
    # current_view = int(seq.split('_')[-1])
    
    # cameraname = join(root_raw, 'cameras', 'rgb_cameras.npz')
    cameraname = cam_path

    cameras = dict(np.load(cameraname))

    novel_view_list = cameras['ids']
    cameras_out = {}
    # for i, cam in enumerate(cameras['ids']):
    #     K = cameras['intrinsics'][i]
    #     dist = cameras['dist_coeffs'][i:i+1]
    #     RT = cameras['extrinsics'][i]
    #     R = RT[:3, :3]
    #     T = RT[:3, 3:]
    #     cameras_out[str(cam)] = {
    #         'K': K,
    #         'dist': dist,
    #         'R': R,
    #         'T': T
    #     }
    #     # 绕x轴转90度
    #     center = - R.T @ T
    #     print(cam, center.T[0])

    # cameras = cameras_out




    shape = np.load(os.path.join(root_data, "mean_shape.npy"))
    num_person = shape.shape[0]
    print("num_person: ", num_person)
    smpl_poses = np.load(os.path.join(root_data, 'poses.npy')) # [::self.skip_step]
    trans = np.load(os.path.join(root_data, 'normalize_trans.npy')) # [::self.skip_step]
    # cameras
    camera_dict = np.load(os.path.join(root_data, 'cameras_normalize.npz'))
    scale_mats = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(smpl_poses.shape[0])] # range(0, self.n_images, self.skip_step)
    world_mats = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(smpl_poses.shape[0])] # range(0, self.n_images, self.skip_step)
    scale = 1 / scale_mats[0][0, 0]
    scale_mat_all = []
    world_mat_all = []
    intrinsics_all = []
    pose_all = []
    for scale_mat, world_mat in zip(scale_mats, world_mats):
        P = world_mat
        # P = world_mat @ scale_mat
        # self.scale_mat_all.append(scale_mat)
        world_mat_all.append(world_mat)
        # self.P.append(P)
        # C = -np.linalg.solve(P[:3, :3], P[:3, 3])
        # self.C.append(C)
        P = P[:3, :4]
        intrinsics, pose = load_K_Rt_from_P(None, P)
        intrinsics_all.append(intrinsics)
        pose_all.append(pose)
    assert len(intrinsics_all) == len(pose_all) # == len(self.images)
    assert (world_mat_all[0] == world_mat_all[-1]).all()

    for novel_view in novel_view_list:
        c_cur = int(np.where(cameras['ids'] == current_view)[0])
        gt_cam_intrinsics_cur = cameras['intrinsics'][c_cur]
        gt_cam_extrinsics_cur = cameras['extrinsics'][c_cur]
        c_tgt = int(np.where(cameras['ids'] == novel_view)[0])
        gt_cam_intrinsics_tgt = cameras['intrinsics'][c_tgt]
        gt_cam_extrinsics_tgt = cameras['extrinsics'][c_tgt]
                                            
        for world_mat in world_mat_all:
            intrinsics_training_cur, pose_training_cur = load_K_Rt_from_P(None, world_mat[:3, :4])
            scale_factor = gt_cam_intrinsics_cur[0, 0] / intrinsics_training_cur[0, 0]
            # print(“scale_factor: “, scale_factor)
            # print(“intrinsics_training_cur: “, intrinsics_training_cur * scale_factor)
            # print(‘target intrinsics’, self.gt_cam_intrinsics_cur)
            R_cur = pose_training_cur[:3, :3].transpose()
            t_cur = -R_cur @ pose_training_cur[:3, 3]
            # print(‘recalculate P’, intrinsics_training_cur[:3, :3] @ np.concatenate((R_cur, t_cur.reshape(3,1)), axis=1))
            # print(‘target P’, world_mat)
            R3 = R_cur
            t3 = t_cur
            R1 = gt_cam_extrinsics_cur[:3, :3]
            t1 = gt_cam_extrinsics_cur[:3, 3]
            Rab = R3.transpose() @ R1
            tab = R3.transpose() @ (t1 - t3)
            R2 = gt_cam_extrinsics_tgt[:3, :3]
            t2 = gt_cam_extrinsics_tgt[:3, 3]
            R4 = R2 @ Rab.transpose()
            t4 = t2 - R4 @ tab
            novel_world_mat = np.eye(4)
            scaled_gt_cam_intrinsics_tgt = gt_cam_intrinsics_tgt[:3, :3].copy()
            scaled_gt_cam_intrinsics_tgt[0, 0] = scaled_gt_cam_intrinsics_tgt[0, 0] / scale_factor
            scaled_gt_cam_intrinsics_tgt[1, 1] = scaled_gt_cam_intrinsics_tgt[1, 1] / scale_factor
            scaled_gt_cam_intrinsics_tgt[0, 2] = scaled_gt_cam_intrinsics_tgt[0, 2] / scale_factor
            scaled_gt_cam_intrinsics_tgt[1, 2] = scaled_gt_cam_intrinsics_tgt[1, 2] / scale_factor
            novel_world_mat[:3, :4] = scaled_gt_cam_intrinsics_tgt @ np.concatenate((R4, t4.reshape(3,1)), axis=1)

            K = scaled_gt_cam_intrinsics_tgt
            R = R4
            T = t4.reshape(3, 1)
            cameras_out[str(novel_view)] = {
                'K': K,
                'dist': np.zeros((1, 5)),
                'R': R,
                'T': T
            }
            break
            











    # meshanme = join(root_data, 'test1.ply')
    meshanme = join(root_data, 'test_1.ply')

    mesh = o3d.io.read_triangle_mesh(meshanme)
    vertices = np.asarray(mesh.vertices)

    R_global = cv2.Rodrigues(np.array([np.pi/2, 0, 0]))[0]
    
    vertices_R = vertices @ R_global.T
    z_min = np.min(vertices_R[:, 2])
    T_global = np.array([0, 0, -z_min]).reshape(3, 1)
    vertices_RT = vertices_R + T_global.T

    # mesh.vertices = o3d.utility.Vector3dVector(vertices_RT)

    # o3d.io.write_triangle_mesh(join(outroot, 'mesh-test.obj'), mesh)

    for key, cam in cameras_out.items():
        cam['R'] = cam['R'] @ R_global.T
        cam.pop('Rvec', '')
        center = - cam['R'].T @ cam['T']
        newcenter = center + T_global
        newT = -cam['R'] @ newcenter
        cam['T'] = newT
        center = - cam['R'].T @ cam['T']
        print(center.T)

    from easymocap.mytools.camera_utils import write_camera
    write_camera(cameras_out, outroot)
    # import pdb;pdb.set_trace()
    mask_list_0 = sorted(glob(join(root_data, 'mask', '0', '*.png')))
    mask_list_1 = sorted(glob(join(root_data, 'mask', '1', '*.png')))
    # label_list_0 = sorted(glob(join(root_raw, 'init_refined_mask', '0', '*.png')))
    # label_list_1 = sorted(glob(join(root_raw, 'init_refined_mask', '1', '*.png')))
    image_list = sorted(glob(join(root_data, 'image', '*.png')))
    for idx in range(smpl_poses.shape[0]):
        params0 = {
            'poses': smpl_poses[idx][0],
            'shapes': shape[0],
            'Th': trans[idx][0]
        }
        params1 = {
            'poses': smpl_poses[idx][1],
            'shapes': shape[1],
            'Th': trans[idx][1]
        }
        outname = join(outroot, 'output-smpl-3d', 'smpl', str(current_view), f'{idx:06d}.json')

        if not os.path.exists(outname) or True:
            params = [params0, params1]
            for i, param in enumerate(params):
                for key in param.keys():
                    param[key] = param[key][None]
                param['Rh'] = np.zeros_like(param['Th'])
                param = body_model.convert_from_standard_smpl(param)

                Rold = cv2.Rodrigues(param['Rh'])[0]
                Told = param['Th']
                Rnew = R_global @ Rold
                Tnew = (R_global @ Told.T + T_global).T
                param['Rh'] = cv2.Rodrigues(Rnew)[0].reshape(1, 3)
                param['Th'] = Tnew.reshape(1, 3)

                param['id'] = i
                params[i] = param
            write_smpl(outname, params)
        os.makedirs(join(outroot, 'output-smpl-3d', 'instance', str(current_view)), exist_ok=True)
        os.makedirs(join(outroot, 'output-keypoints3d', 'label', str(current_view)), exist_ok=True)
        os.makedirs(join(outroot, 'images', str(current_view)), exist_ok=True)
        
        shutil.copyfile(mask_list_0[idx], join(outroot, 'output-smpl-3d', 'instance', str(current_view), f'{idx:06d}_0.png'))
        shutil.copyfile(mask_list_1[idx], join(outroot, 'output-smpl-3d', 'instance', str(current_view), f'{idx:06d}_1.png'))
        shutil.copyfile(image_list[idx], join(outroot, 'images', str(current_view), f'{idx:06d}.png'))
        # shutil.copyfile(label_list_0[idx], join(outroot, 'output-keypoints3d', 'label', str(current_view), f'{idx:06d}_0.png'))
        # shutil.copyfile(label_list_1[idx], join(outroot, 'output-keypoints3d', 'label', str(current_view), f'{idx:06d}_1.png'))
        shutil.copyfile('/media/ubuntu/hdd/easymocap/EasyMocap/data/soccer_single/output-smpl-3d/cfg_model.yml', join(outroot, 'output-smpl-3d', 'cfg_model.yml'))

    vertices = body_model(**param)[0]
    # import pdb;pdb.set_trace()
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices.reshape(-1, 3).cpu().numpy())
    mesh.triangles = o3d.utility.Vector3iVector(body_model.faces.reshape(-1, 3))
    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(join(outroot, 'mesh-test.obj'), mesh)
    
    # extract vertices from smpl
    # python3 apps/postprocess/write_vertices.py ${data}/output-smpl-3d/smpl ${data}/output-smpl-3d/vertices --cfg_model ${data}/output-smpl-3d/cfg_model.yml --mode vertices

    # if not os.path.exists(join(outroot, 'images')):
    #     shutil.copytree(join(root_data, 'image'), join(outroot, 'images', str(current_view)))

    # regressor = np.load('data/smplx/J_regressor_body25.npy')
    # filenames = sorted(glob(join(root, 'smpl', '*.npz')))
    # for filename in tqdm(filenames):
    #     data = dict(np.load(filename))
    #     vertices = data['verts']
    #     vertices = vertices @ R_global.T + T_global.T
    #     joints = np.matmul(regressor[None], vertices)
    #     # 绕x轴旋转90度
    #     results = [{
    #         'id': 0,
    #         'keypoints3d': joints[0]
    #     },
    #     {
    #         'id': 1,
    #         'keypoints3d': joints[1]
    #     }]
    #     outname = join(outroot, 'body25', os.path.basename(filename).replace('.npz', '.json'))
    #     if not os.path.exists(outname):
    #         write_keypoints3d(outname, results)
    #     results = [{
    #         'id': 0,
    #         'vertices': vertices[0]
    #     },
    #     {
    #         'id': 1,
    #         'vertices': vertices[1]
    #     }]
    #     outname = join(outroot, 'vertices-gt', os.path.basename(filename).replace('.npz', '.json'))
    #     if not os.path.exists(outname):
    #         write_vertices(outname, results)
    #     params0 = {
    #         'poses': np.hstack([data['global_orient'][0], data['body_pose'][0]]),
    #         'shapes': data['betas'][0],
    #         'Th': data['transl'][0]
    #     }
    #     params1 = {
    #         'poses': np.hstack([data['global_orient'][1], data['body_pose'][1]]),
    #         'shapes': data['betas'][1],
    #         'Th': data['transl'][1]
    #     }
    #     outname = join(outroot, 'smpl-gt', os.path.basename(filename).replace('.npz', '.json'))
    #     if not os.path.exists(outname) or True:
    #         params = [params0, params1]
    #         for i, param in enumerate(params):
    #             for key in param.keys():
    #                 param[key] = param[key][None]
    #             param['Rh'] = np.zeros_like(param['Th'])
    #             param = body_model.convert_from_standard_smpl(param)
    #             Rold = cv2.Rodrigues(param['Rh'])[0]
    #             Told = param['Th']
    #             Rnew = R_global @ Rold
    #             Tnew = (R_global @ Told.T + T_global).T
    #             param['Rh'] = cv2.Rodrigues(Rnew)[0].reshape(1, 3)
    #             param['Th'] = Tnew.reshape(1, 3)
    #             param['id'] = i
    #             params[i] = param
    #         write_smpl(outname, params)
    # if not os.path.exists(join(outroot, 'images')):
    #     shutil.copytree(join(root, 'images'), join(outroot, 'images'))