import torch
import torch.utils.data
import cv2
import numpy as np
import torch.nn.functional as F
import os
import glob

# Note:
# You should be only concerned with the following files:
# IBUG_x.png : This is your image (256, 256, 3)
# IBUG_x_frontalized_new.npy : This is global scaled position map aka mesh (256, 256, 3)
# IBUG_x_procrustes_tform_new.npy : Think of this as your camera matrix (See the load camera matrix function below to load it properly)
# IBUG_x_top_down.txt : This is your top_down map which is of format (256, 256, 2). Basically, given (y, x) on your image, top_down(y, x) will return (v, u)
# y, x and v, u are in range 0 to 255
# To understand the reprojection cycle refer to the debug section at the bottom (you need to be careful there is a y-flip which could be easily forgotten).

# If you run this script just as it is, it will load everything and keep on doing one reprojection loop per image
# After reprojection there could be an error of +/- 1 pixels in x or y or both due to int operation

INPUT_IMG_SCALE = 1.0
SHOULD_OVERFIT = False
LOGLEVEL = 'INFO'
SUPERVISION = {
    'visibility': True,
}


class DepthGenerator(object):
    """
    Generates depth.
    """
    def __init__(self, ):
        self.face_ind = np.loadtxt('./misc/face_ind_faces.txt').astype(
            np.int32)
        # Load triangles
        triangles = np.loadtxt('./misc/triangles_faces.txt').astype(np.int32)
        self.triangles = triangles.T
        self.preload_mean_posmap()
        self.preload_true_mask()

    @staticmethod
    def load_top_down(path):
        '''
        Args:
            path: path to the top down binary file 
        Returns:
            top_down_map: shape will be (256, 256, 2) dtype: int
        '''
        return np.fromfile(path, dtype=int).reshape(256, 256, 2)

    @staticmethod
    def load_camera_matrix(path):
        '''
        Args:
            path: path to the procrustes_tfrom_new.npy file
        Returns:
            top_down_map: shape will be 3x4 float matrix dtype:float64
        '''

        # 3x4 RTS matrix which converts vertices in world to frontalized vertices
        # We need frontalized vertices to vertices in world hence we take inverse
        procrustes_tform_3x4 = np.load(path)

        # Make 3x4 to 4x4
        procrustes_tform_4x4 = np.vstack([procrustes_tform_3x4, [0, 0, 0, 1]])

        # Compute its inverse
        inverse_tform_4x4 = np.linalg.inv(procrustes_tform_4x4)

        # Make 4x4 to 3x4
        inverse_tform_3x4 = np.delete(inverse_tform_4x4, 3, 0)

        return inverse_tform_3x4

    @staticmethod
    def isPointInTri(point, tri_points):
        ''' Judge whether the point is in the triangle
        Method:
            http://blackpawn.com/texts/pointinpoly/
        Args:
            point: [u, v] or [x, y] 
            tri_points: three vertices(2d points) of a triangle. 2 coords x 3 vertices
        Returns:
            bool: true for in triangle
        '''
        tp = tri_points

        # vectors
        v0 = tp[:, 2] - tp[:, 0]
        v1 = tp[:, 1] - tp[:, 0]
        v2 = point - tp[:, 0]

        # dot products
        dot00 = np.dot(v0.T, v0)
        dot01 = np.dot(v0.T, v1)
        dot02 = np.dot(v0.T, v2)
        dot11 = np.dot(v1.T, v1)
        dot12 = np.dot(v1.T, v2)

        # barycentric coordinates
        if dot00 * dot11 - dot01 * dot01 == 0:
            inverDeno = 0
        else:
            inverDeno = 1 / (dot00 * dot11 - dot01 * dot01)

        u = (dot11 * dot02 - dot01 * dot12) * inverDeno
        v = (dot00 * dot12 - dot01 * dot02) * inverDeno

        # check if point in triangle
        return (u >= 0) & (v >= 0) & (u + v < 1)

    @staticmethod
    def barycentric(point, tri_points):
        ''' Judge whether the point is in the triangle
        Method:
            http://blackpawn.com/texts/pointinpoly/
        Args:
            point: [u, v] or [x, y] 
            tri_points: three vertices(2d points) of a triangle. 2 coords x 3 vertices
        Returns:
            bool: true for in triangle
        '''
        tp = tri_points.copy()

        # vectors
        # tp[:,2] = b v
        # tp[:,0] = a u
        # tp[:,1] = c w

        v0 = tp[:, 2] - tp[:, 0]
        v1 = tp[:, 1] - tp[:, 0]
        v2 = point - tp[:, 0]

        # dot products
        dot00 = np.dot(v0.T, v0)
        dot01 = np.dot(v0.T, v1)
        dot02 = np.dot(v0.T, v2)
        dot11 = np.dot(v1.T, v1)
        dot12 = np.dot(v1.T, v2)

        # barycentric coordinates
        if dot00 * dot11 - dot01 * dot01 == 0:
            inverDeno = 0
        else:
            inverDeno = 1 / (dot00 * dot11 - dot01 * dot01)

        v = (dot11 * dot02 - dot01 * dot12) * inverDeno
        w = (dot00 * dot12 - dot01 * dot02) * inverDeno
        u = 1.0 - (v + w)
        # check if point in triangle
        return u, w, v

    def get_vertices(self, pos):
        '''
        Args:
            pos: the 3D position map. shape = (256, 256, 3).
        Returns:
            vertices: the vertices(point cloud). shape = (num of points, 3). n is about 40K here.
        '''
        resolution_op = 256
        #         face_ind = np.loadtxt('face_ind.txt').astype(np.int32)
        all_vertices = np.reshape(pos, [resolution_op**2, -1])
        vertices = all_vertices[self.face_ind, :]
        return vertices.T

    def rasterize_mesh(self, vertices):
        '''
        Args:
            vertices: Vertices of mesh in image space (65k, 3)
        Returns:
            depth_map: Depth map obtained by rasterization  (256, 256, 3) float
        '''

        vertices = self.get_vertices(vertices)
        triangles = self.triangles
        frontalized_map = self.mean_pmap

        h = 256
        w = 256
        # Average depth for initialization
        tri_depth = (vertices[2, triangles[0, :]] +
                     vertices[2, triangles[1, :]] +
                     vertices[2, triangles[2, :]]) / 3.

        # Required depth map
        depth_map = np.zeros(
            (frontalized_map.shape[0], frontalized_map.shape[1], 3), np.float)
        depth_map[:, :, :] = 0

        # Set texture width and height
        tw = 256
        th = 256

        depth_buffer = np.zeros([h, w]) - 999999.
        for i in range(triangles.shape[1]):
            # Note you don't need to do -1 since trinagles.txt already has it
            tri = triangles[:, i]  # 3 vertex indices

            # the inner bounding box
            umin = max(int(np.ceil(np.min(vertices[0, tri]))), 0)
            umax = min(int(np.floor(np.max(vertices[0, tri]))), w - 1)

            vmin = max(int(np.ceil(np.min(vertices[1, tri]))), 0)
            vmax = min(int(np.floor(np.max(vertices[1, tri]))), h - 1)

            if umax < umin or vmax < vmin:
                continue

            for u in range(umin, umax + 1):
                for v in range(vmin, vmax + 1):
                    # Only if the point is in the triangle and its depth value is greater than depth buffer
                    # we reassign it as depth buffer
                    if tri_depth[i] > depth_buffer[v, u] and self.isPointInTri(
                        [u, v], vertices[:2, tri]):
                        depth_buffer[v, u] = tri_depth[i]
                        alpha, beta, gamma = self.barycentric([u, v],
                                                              vertices[:2,
                                                                       tri])

                        xyz = (alpha * vertices[:, tri[0]] +
                               beta * vertices[:, tri[1]] + gamma *
                               vertices[:, tri[2]]) / (alpha + beta + gamma)

                        depth_map[v, u, :] = xyz[2]

        return depth_map

    def preload_mean_posmap(self, ):
        mean_posmap_path = './misc/face_mean_posmap.png'
        mean_pmap = cv2.imread(mean_posmap_path, cv2.IMREAD_UNCHANGED).astype(
            np.float32) / 255.0
        self.mean_pmap = mean_pmap * 2 - 255.0

    def preload_true_mask(self, ):
        true_fmask_path = './misc/true_face_mask.png'
        mask = cv2.imread(true_fmask_path, cv2.IMREAD_UNCHANGED)
        mask[mask > 0] = 255
        self.mask = mask

    def get_depth_map(self, cam_path, front_path=None, img_size=(256, 256)):
        # Pic path and depth path can be inferred from cam path.
        depth_path = cam_path.replace('_procrustes_tform_new.npy',
                                      '_depth.png')
        if os.path.isfile(depth_path):
            # If we have it cached, return it:
            depth_map = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            depth_map = cv2.resize(depth_map, img_size)
            depth_map = depth_map.astype(np.float32) / 255.0

        else:
            # Generate it and cache it.
            depth_map = self.render_depth_map(cam_path)
            to_write = (depth_map * 255).astype(np.uint16)
            cv2.imwrite(depth_path, to_write)
        # Need 256, 256, 1 shape out.
        return depth_map

    def render_depth_map(self, cam_path, front_path=None):
        # Load the position map and procrustes tform (Shape is [256, 256, 3])
        if front_path is not None:
            frontalized_map = np.load(front_path)
        else:
            frontalized_map = self.mean_pmap

        # Load camera matrix which converts vertices in frontalized frame to camera space (i.e 3D PRNet format)
        camera_matrix = self.load_camera_matrix(cam_path)

        # Convert to (65k, 3)
        frontalized_map_reshaped = np.reshape(frontalized_map.copy(), [-1, 3])

        # Apply camera matrix
        vertices_homogeneous = np.hstack(
            (frontalized_map_reshaped.copy(),
             np.ones([frontalized_map_reshaped.shape[0], 1])))
        vertices_image = vertices_homogeneous.dot(camera_matrix.T)

        # Reshape to [256, 256, 3]
        vertices_image = np.reshape(vertices_image, [256, 256, 3])

        # Remove outliers (Will be present since the vertices are 65k and not 43k)
        vertices_image[vertices_image > 255] = 255
        # Invert y
        vertices_image[:, :, 1] = 256 - 1 - vertices_image[:, :, 1]

        # Apply mask (will make non head portions 0)
        vertices_image = vertices_image * self.mask
        vertices_image = vertices_image / 255

        # Now we only have vertices in image space but to get depth map, one needs to rasterize these vertices using triangle faces

        # Convert vertices back to (65k, 3)
        vertices_image_reshaped = np.reshape(vertices_image.copy(), [-1, 3])
        print('vertices_image_reshaped min, max: {}, {}'.format(
            np.min(vertices_image_reshaped[:, 2]),
            np.max(vertices_image_reshaped[:, 2])))

        # Compute depth map by rasterizing mesh (Note depth is float)
        depth_map = self.rasterize_mesh(vertices_image_reshaped)
        print("depth_map min/max: {}/{}".format(np.min(depth_map),
                                                np.max(depth_map)))
        return depth_map[..., 2]


class Face_instance_holder(object):
    def __init__(self):
        self.instances = {}

    def add(self, filepath):
        assert '300W_LP' in filepath, 'Face Instance holder should get 300W_LP filepaths!'
        filename = filepath.split('/')[-1]
        num_underscores = filename.count('_')
        if num_underscores == 5 or num_underscores == 6:
            instance_id = '_'.join(filename.split('_')[:-3])
            # Specific to dataset structure:
            prepend = filename.split('_')[0]
            instance_id = prepend + '/' + instance_id
        else:
            print("Num underscores found {}, while expected either 5 or 6".
                  format(num_underscores))
        if instance_id not in self.instances:
            self.instances[instance_id] = Face_instance(filename)
        self.instances[instance_id].update_pic_ids(filename)

    def __call__(self, instance_id, pic_id, prepend=None):
        return self.instances[instance_id](pic_id, prepend=prepend)


class Face_instance(object):
    def __init__(self, filename):
        # Paths to attributes of each pic
        self.pics = {}
        self.posmap = filename

    def update_pic_ids(self, filename):
        pic_id = filename.split('_')[-3]
        self.pics[pic_id] = filename

    def __call__(self, pic_id, prepend=None):
        # Generates a dict of paths from a single path saved:
        seg_mask_path = self.pics[pic_id]
        dset_prepend = seg_mask_path.split('_')[0]
        seg_mask_path = dset_prepend + '/' + seg_mask_path
        if prepend is not None:
            seg_mask_path = os.path.join(prepend, seg_mask_path)
        pic_dict = {}
        pic_dict['seg_mask'] = seg_mask_path
        pic_dict['image'] = seg_mask_path.replace('_seg_mask.png', '.png')
        pic_dict['frontalized'] = seg_mask_path.replace(
            'seg_mask.png', 'frontalized_new.npy')
        pic_dict['procrustes'] = seg_mask_path.replace(
            'seg_mask.png', 'procrustes_tform_new.npy')
        pic_dict['top_down'] = seg_mask_path.replace('seg_mask.png',
                                                     'top_down.png')
        return pic_dict

        # Attributes:
        """
        'IBUG_image_003_1_0.png',
        'IBUG_image_003_1_0_frontalized_new.npy',
        'IBUG_image_003_1_0_posmap.png',
        'IBUG_image_003_1_0_procrustes_tform_new.npy',
        'IBUG_image_003_1_0_seg_mask.png',
        'IBUG_image_003_1_0_top_down.txt',
        """


class FaceDataset(torch.utils.data.Dataset):
    def __init__(self,
                 root_dir,
                 train_val_test,
                 size=(256, 256),
                 depth_renderer_cache=None,
                 sets_to_use=None,
                 **kwargs):

        self.root_dir = root_dir
        if train_val_test == 'test':
            train_val_test = 'val'
        if not (train_val_test in ["train", "val", "test"]):
            print("train_val_test must be train, val or test. Exiting.")
            exit()
        self.subset = train_val_test
        if sets_to_use is None:
            train_sets = ['LFPW', 'HELEN', 'IBUG']
            val_set = ['AFW']
            if SHOULD_OVERFIT:
                train_sets = ['Overfit']
                val_set = ['Overfit']
            self.sets = train_sets if self.subset == 'train' else val_set
        else:
            self.sets = sets_to_use

        self.load_subset_pic_ids_and_labels()
        mask_path = './misc/true_face_mask.png'
        mask_weight = cv2.imread(mask_path)
        mask_weight = torch.Tensor(mask_weight)
        mask_weight[mask_weight > 0.0] = 1.0
        try:
            self.posmap_mask = ~mask_weight[
                ..., 0].bool()  # Invert to nullify background.
        except AttributeError as e:
            # Supporting PT 1.1, it doesn't have bool:
            print(e)
            self.posmap_mask = ~mask_weight[
                ..., 0].byte()  # Invert to nullify background.
        if SUPERVISION['visibility']:
            self.depth_renderer_cache = DepthGenerator()

    @staticmethod
    def sort_keys(x):
        # Sorting keying function:
        # First arg is the instance_id, second arg is the int of pic_id:
        return ('_'.join(x.split('_')[:-3]), int(x.split('_')[-3]))

    def load_subset_pic_ids_and_labels(self):
        sets = self.sets
        assert 'IBUG' in os.listdir(
            self.root_dir
        ), "Root dir should contain at least the IBUG dataset for training!"
        filenames = []
        for s in sets:
            filenames += glob.glob(self.root_dir + s + '/*_seg_mask.png',
                                   recursive=True)
        face_holder = Face_instance_holder()
        for filepath in filenames:
            face_holder.add(filepath)
        fnames = [x.split('/')[-1] for x in filenames
                  ]  # Store only the important bit, drop the commond part.
        # Sorting keying function:
        # First arg is the instance_id, second arg is the int of pic_id:
        self.fnames = sorted(fnames, key=self.sort_keys)
        self.face_holder = face_holder
        size = 0
        for name, instance in face_holder.instances.items():
            size += len(instance.pics)
        self.size = size

        # Preloading weight mask here to avoid rereading:
        mask_path = './misc/true_face_mask.png'
        mask_weight = cv2.imread(mask_path)
        self.mask_weight = torch.Tensor(np.array(mask_weight, np.float32))

    def __len__(self):
        return self.size

    def idx_to_ids(self, idx):
        filename = self.fnames[idx]
        instance_id = '_'.join(filename.split('_')[:-3])
        # Specific to dataset structure:
        prepend = filename.split('_')[0]
        instance_id = prepend + '/' + instance_id
        pic_id = filename.split('_')[-3]
        return instance_id, pic_id

    @staticmethod
    def load_top_down(path, img_size):
        '''
        Args:
            path: path to the top down png image
        Returns:
            top_down_map: shape will be (256, 256, 2) dtype: int
        '''
        td_png = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        td_png = cv2.resize(td_png, img_size)
        td_png = torch.Tensor(td_png).float()[..., :2] / 255.0
        td_png[td_png == 1.0] = -1.0
        return td_png

    @staticmethod
    def load_camera_matrix(path):
        '''
        Args:
            path: path to the procrustes_tfrom_new.npy file
        Returns:
            top_down_map: shape will be 3x4 float matrix dtype:float64
        '''

        # 3x4 RTS matrix which converts vertices in world to frontalized vertices
        # We need frontalized vertices to vertices in world hence we take inverse
        procrustes_tform_3x4 = np.load(path)

        # Make 3x4 to 4x4
        procrustes_tform_4x4 = np.vstack([procrustes_tform_3x4, [0, 0, 0, 1]])

        # Compute its inverse
        inverse_tform_4x4 = np.linalg.inv(procrustes_tform_4x4)

        return inverse_tform_4x4

    def __getitem__(self, idx):
        # Need mapping from linear index to instance ids and then pic ids within the instance.
        # idx is linear in range [0,self.size]
        data_dict = {}  # Data should be 1CHW Image, etc. All torch.Tensors.
        instance_id, pic_id = self.idx_to_ids(idx)
        path_dict = self.face_holder(instance_id,
                                     pic_id,
                                     prepend=self.root_dir)
        # Need to prepend subset_dir to the paths in this dict ^

        image = cv2.imread(path_dict['image'])
        width = int(image.shape[1] * INPUT_IMG_SCALE)
        height = int(image.shape[0] * INPUT_IMG_SCALE)
        self.img_size = (width, height)
        image = cv2.resize(image, self.img_size)
        if image is None:
            print(path_dict['image'], "is None! Returning next item!")
            return self.__getitem__(idx + 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = torch.Tensor(image).float().permute(2, 0,
                                                    1) / 255.0  # HWC --> CHW

        # Load the position map and procrustes tform (Shape is [256, 256, 3])
        posmap = np.load(path_dict['frontalized'])
        posmap = np.array(posmap, dtype=np.float32)
        posmap = posmap / 255.0  # Need to normalize this for NN training, as it's done in [0,1] range.
        # Cap at 1.0
        posmap[posmap > 1.0] = 1.0
        posmap = torch.Tensor(posmap).float().permute(2, 0, 1)  # HWC --> CHW
        # Will mask out posmap:
        posmap[
            ...,
            self.posmap_mask] = 1.0  # Mask out neck as we don't consider it.

        # Load camera matrix which converts vertices in frontalized frame to camera space (i.e 3D PRNet format)
        camera_pose = torch.Tensor(
            self.load_camera_matrix(path_dict['procrustes']))
        camera_pose = self.make_single_pose_tensor(camera_pose.float(),
                                                   torch.eye(3, 3),
                                                   torch.zeros(4))
        if SUPERVISION['visibility']:
            cam_path = path_dict['procrustes']
            depth = self.depth_renderer_cache.get_depth_map(
                cam_path, img_size=self.img_size)
            depth = torch.Tensor(depth).unsqueeze(2)
            data_dict['depths'] = depth

        # Load segmentation mask
        seg_mask = cv2.imread(path_dict['seg_mask'])
        seg_mask = cv2.resize(seg_mask, self.img_size)
        seg_mask = self.prepare_segmask(
            torch.Tensor(seg_mask).float() / 255.0)  # DIMS??

        # Load top_down_map
        top_down_map = self.load_top_down(
            path_dict['top_down'], self.img_size).permute(2, 0,
                                                          1)  # HWC --> CHW

        data_dict['imgs'] = image
        data_dict['uv_labels'] = top_down_map
        data_dict['cls_labels'] = seg_mask
        data_dict['quat_labels'] = torch.ones(1, 4)  # Fake it
        data_dict['posmap_labels'] = posmap
        data_dict['poses'] = camera_pose
        return data_dict

    @staticmethod
    def prepare_segmask(seg_mask):
        class_ids = seg_mask[..., 0]
        # Need to switch background and foreground to comply with the dataset format:
        class_ids = (class_ids - 1) * -1
        class_ids = class_ids.long()
        one_hot = torch.nn.functional.one_hot(class_ids, 2)  # size=(h,w,2)
        one_hot = one_hot.permute(2, 0, 1).float()
        return one_hot

    @staticmethod
    def make_single_pose_tensor(Rt, K, dist):
        """Converts 3 camera pose matrices into one to avoid cluttering up the code

        Arguments:
            Rt {torch.Tensor} -- 4x4 Rt matrix
            K {torch.Tensor} -- 3x3 Intrinsics matrix
            dist {torch.Tensor} -- 4 distortion vector

        Returns:
            torch.Tensor -- 8x4 pose tensor containing all three tensors concat together
        """
        K = F.pad(K.clone(), [0, 1, 0, 0])  # Add the 4th column to intrinsics
        dist = dist.clone().unsqueeze(0)
        pose = torch.cat([Rt, K, dist])
        return pose

    @staticmethod
    def read_dict_to_tensor_dict(pose_dict):
        tensor_dict = {}
        for frame_num, rtkdist in pose_dict.items():
            tensor_dict[int(frame_num)] = {}
            for rt_or_k_or_dist, list_array in rtkdist.items():
                tensor_dict[int(frame_num)][rt_or_k_or_dist] = torch.Tensor(
                    list_array)
        return tensor_dict


class FacePairDataset(FaceDataset):
    @staticmethod
    def parse_instance_id_pic_id(_id):
        assert len(_id.split(
            "/")) == 2, "Inconsistent state while parsing: {}".format(_id)
        instance_id, pic_id = _id.split("/")[0], _id.split("/")[1]
        return instance_id, pic_id

    def __init__(
            self,
            root_dir,
            train_val_test,
            size=(256, 512),
            depth_renderer_cache=None,
    ):
        super(FacePairDataset, self).__init__(
            root_dir=root_dir,
            train_val_test=train_val_test,
            size=size,
            depth_renderer_cache=depth_renderer_cache,
        )

    def __getitem__(self, idx):

        # NOTE: Can choose pairs on random as well i.e. idx + randint(1, 3).
        # Shuffle is done by giving in random IDX.
        # Hence we need to take the original IDX and return another nearby pic for the same instance
        prev, current, nxt = idx - 1, idx, idx + 1
        # Need to know how many pic ids there are for each instance.
        # Or just parse the next idx and have the fnames array sorted, so that same fnames are close by
        # Then index the next idx, check if it's from the same instance.
        # If not, get the previous idx.

        curr_instance_id, curr_pic_id = self.idx_to_ids(current)
        if nxt >= self.size:
            # At the end of the dataloader, sample previous pic_id.
            nxt = prev
        nxt_instance_id, nxt_pic_id = self.idx_to_ids(nxt)
        if nxt_instance_id != curr_instance_id:
            # Case when we're on the last pic of some instance: take a prev instance.
            # Sample prev idx then.
            prev_instance_id = nxt_instance_id  # For debugging
            nxt = prev  # Swap them to keep logic simpler down below.
            nxt_instance_id, nxt_pic_id = self.idx_to_ids(nxt)
        if nxt_instance_id != curr_instance_id:
            err_msg = "Instance ids don't match!: {} != {}; {}".format(
                nxt_instance_id, curr_instance_id, prev_instance_id)
            print(err_msg, 'idx = ', idx)
            nxt = current
        data_1 = super().__getitem__(current)  # Calling Parent's method
        data_2 = super().__getitem__(nxt)
        return data_1, data_2
