import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

def normalize(v):
    """Normalize a vector."""
    return v/np.linalg.norm(v)

def average_poses(poses):
    """
    Calculate the average pose, which is then used to center all poses
    using @center_poses. Its computation is as follows:
    1. Compute the center: the average of pose centers.
    2. Compute the z axis: the normalized average z axis.
    3. Compute axis y': the average y axis.
    4. Compute x' = y' cross product z, then normalize it as the x axis.
    5. Compute the y axis: z cross product x.
    
    Note that at step 3, we cannot directly use y' as y axis since it's
    not necessarily orthogonal to z axis. We need to pass from x to y.

    Inputs:
        poses: (N_images, 3, 4)

    Outputs:
        pose_avg: (3, 4) the average pose
    """
    # 1. Compute the center
    center = poses[..., 3].mean(0) # (3)

    # 2. Compute the z axis
    z = normalize(poses[..., 2].mean(0)) # (3)

    # 3. Compute axis y' (no need to normalize as it's not the final output)
    y_ = poses[..., 1].mean(0) # (3)

    # 4. Compute the x axis
    x = normalize(np.cross(y_, z)) # (3)

    # 5. Compute the y axis (as z and x are normalized, y is already of norm 1)
    y = np.cross(z, x) # (3)

    pose_avg = np.stack([x, y, z, center], 1) # (3, 4)

    return pose_avg

def center_poses(poses):
    """
    Center the poses so that we can use NDC.
    See https://github.com/bmild/nerf/issues/34

    Inputs:
        poses: (N_images, 3, 4)

    Outputs:
        poses_centered: (N_images, 3, 4) the centered poses
    """

    pose_avg = average_poses(poses) # (3, 4)
    pose_avg_homo = np.eye(4)
    pose_avg_homo[:3] = pose_avg # convert to homogeneous coordinate for faster computation
                                 # by simply adding 0, 0, 0, 1 as the last row
    last_row = np.tile(np.array([0, 0, 0, 1]), (len(poses), 1, 1)) # (N_images, 1, 4)
    poses_homo = \
        np.concatenate([poses, last_row], 1) # (N_images, 4, 4) homogeneous coordinate

    poses_centered = np.linalg.inv(pose_avg_homo) @ poses_homo # (N_images, 4, 4)
    poses_centered = poses_centered[:, :3] # (N_images, 3, 4)

    return poses_centered

def create_spiral_poses(original_poses, radii, n_poses=180):
    """
    Create spiral (or not) poses around given poses for novel view rendering purpose.
    Inputs:
        original_poses: (#N, 3, 4) original poses around which to generate the spiral.
        radii: (3) radii of the spiral for each axis (only x, y are used)
        n_poses: int, number of poses to create
    Outputs:
        poses_spiral: (#n_poses, 3, 4) the poses in the spiral path
    """
    N_frames = len(original_poses)
    # interpolation rotations
    rot_slerp = Slerp(range(N_frames), R.from_matrix(original_poses[..., :3]))  # rot_slerp is a generator
    interp_rots = rot_slerp(np.linspace(0, N_frames-1, n_poses+1))[:-1]
    interp_rots = interp_rots.as_matrix() # (#n_poses, 3, 3)
    # interpolation positions
    interp_xyzs = np.stack([np.interp(np.linspace(0, N_frames-1, n_poses+1)[:-1], range(N_frames),
                                      original_poses[:, i, 3]) for i in range(3)], -1) # (#n_poses, 3)

    poses_spiral = []
    for i, t in enumerate(np.linspace(0, 8*np.pi, n_poses+1)[:-1]): # rotate 8pi (4 rounds)
        pose = np.zeros((3, 4))
        pose[:, :3] = interp_rots[i]
        pose[:, 3] = interp_xyzs[i] + radii * np.array([np.cos(t), -np.sin(t), 0])
        poses_spiral += [pose]

    return np.stack(poses_spiral, 0) # (#n_poses, 3, 4)


def create_spiral_poses_single(c2w, max_trans, n_poses=60):
    """
        Create spiral poses around one pose for novel view rendering purpose.
        Inputs:
            c2w: (3,4)
            max_trans: scalar
            n_poses: int, number of poses to create
        Outputs:
            poses_spiral: (#n_poses, 3, 4) the poses in the spiral path    
    """
    output_poses = []

    for i in range(n_poses):
        x_trans = max_trans * np.sin(2.0 * np.pi * float(i) / float(n_poses))
        y_trans = max_trans * np.cos(2.0 * np.pi * float(i) / float(n_poses))/2.0
        z_trans = 0
        #z_trans = max_trans * np.cos(2.0 * np.pi * float(i) / float(n_poses))

        cur_pos = c2w.copy()
        cur_pos[:,3] += np.array([x_trans, y_trans, z_trans]).reshape(3,)

        output_poses.append(cur_pos)
    
    return np.stack(output_poses, 0) # (#n_poses, 3, 4)
