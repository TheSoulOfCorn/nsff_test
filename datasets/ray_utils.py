import torch
from einops import rearrange
import numpy as np
from kornia import create_meshgrid


def get_ray_directions(H, W, K, D=None, return_uv=False, flatten=True):
    """
    Get ray directions for all pixels in camera coordinate.
    Inputs:
        H, W: image height and width
        K: (3, 3) camera intrinsics
        D: None or (M) radial distortion parameters
        return_uv: whether to return uv image coordinates

    Outputs: (shape depends on @flatten)
        directions: (H, W, 3) the direction of the rays in camera coordinate
        uv: (H, W, 2) image coordinates
    """
    grid = create_meshgrid(H, W, normalized_coordinates=False)[0] # (H, W, 2)
    i, j = grid.unbind(-1)  # (H,W) loc, i ~ x , j ~ y

    # without +0.5 pixel centering
    # u/i = fx *X + cx  --> X = (i-cx)/fx
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    # cx: W/2 
    # cy: H/2
    # scan line from the left top to right bot
    directions = torch.stack([(i-cx)/fx, -(j-cy)/fy, -torch.ones_like(i)], -1)
    if flatten:
        directions = directions.reshape(-1, 3)
        grid = grid.reshape(-1, 2)
    if return_uv:
        return directions, grid
    return directions

def get_rays(directions, c2w):
    """
    Get ray origin and normalized directions in world coordinate for all pixels in one image.
    Inputs:
        directions: (HW, 3) precomputed ray directions in camera coordinate
        c2w: (3, 4) transformation matrix from camera coordinate to world coordinate
    Outputs:
        rays_o: (HW, 3), the origin of the rays in world coordinate
        rays_d: (HW, 3), the normalized direction of the rays in world coordinate
    """
    # Rotate ray directions from camera coordinate to the world coordinate
    rays_d = directions @ c2w[:, :3].T # (HW, 3)
    rays_d /= torch.norm(rays_d, dim=-1, keepdim=True)
    # The origin of all rays is the camera origin in world coordinate
    rays_o = c2w[:, 3].expand_as(rays_d) # (HW, 3)

    return rays_o, rays_d

def get_ndc_rays(K, near, shift_near, rays_o, rays_d):
    """
    Transform rays from world coordinate to NDC.
    https://github.com/bmild/nerf/files/4451808/ndc_derivation.pdf

    Inputs:
        K: (3, 3) camera intrinsics
        near: (N_rays) or float, the depths of the near plane
        shift_near: (N_rays) or float, the amount to shift the origin
        rays_o: (N_rays, 3), the origin of the rays in world coordinate
        rays_d: (N_rays, 3), the direction of the rays in world coordinate

    Outputs:
        rays_o: (N_rays, 3), the origin of the rays in NDC
        rays_d: (N_rays, 3), the direction of the rays in NDC
    """
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    # Shift ray origins to near plane
    t = -(shift_near + rays_o[...,2]) / rays_d[...,2]
    rays_o = rays_o + t[...,None] * rays_d

    # Store some intermediate homogeneous results
    ox_oz = rays_o[...,0] / rays_o[...,2]
    oy_oz = rays_o[...,1] / rays_o[...,2]
    
    # Projection
    o0 = -1./(cx/fx) * ox_oz
    o1 = -1./(cy/fy) * oy_oz
    o2 = 1. + 2. * near / rays_o[...,2]

    d0 = -1./(cx/fx) * (rays_d[...,0]/rays_d[...,2] - ox_oz)
    d1 = -1./(cy/fy) * (rays_d[...,1]/rays_d[...,2] - oy_oz)
    d2 = 1 - o2
    
    rays_o = torch.stack([o0, o1, o2], -1) # (B, 3)
    rays_d = torch.stack([d0, d1, d2], -1) # (B, 3)
    
    return rays_o, rays_d

# not used
def world2ndc(xyz, K):
    """
    Convert world coordinates into NDC coordinates.
    Inputs:
        xyz: (N, 3) world coordinates
        K: (3, 3) camera intrinsics

    Outputs:
        ndc: (N, 3) ndc coordinates
    """
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    NDC_x = -fx/cx*xyz[:, 0]/xyz[:, 2]
    NDC_y = -fy/cy*xyz[:, 1]/xyz[:, 2]
    NDC_z = 1+2/xyz[:, 2]
    ndc = rearrange([NDC_x, NDC_y, NDC_z], 'c n -> n c', c=3)
    return ndc

def ndc2world(xyz, K, eps=1e-6):
    """
    Convert NDC coordinates into world coordinates.
    DO NOT mess this with ndc_rays generation
    Inputs:
        xyz: (N, 3) or (N, M, 3) NDC coordinates
        K: (N,3,3) camera intrinsics
        eps: float to prevent division by zero
    Outputs:
        world: (N, 3) world coordinates
    """
    fx, fy, cx, cy = K[..., 0, 0], K[..., 1, 1], K[..., 0, 2], K[..., 1, 2]  # (N)

    Rz = 2/(xyz[..., 2]-1-eps)  # (N) or (N,M)
    if len(xyz.shape) == 2: # case for xyz_fw, weighted, all dim (N)
        Rx = -Rz*xyz[..., 0]*cx/fx
        Ry = -Rz*xyz[..., 1]*cy/fy
        world = rearrange([Rx, Ry, Rz], 'c n -> n c', c=3)  # (N,3)
    elif len(xyz.shape) == 3: # case for xyzs_fw, not weighted, all dim (N,M)
        Rx = -Rz*xyz[..., 0]*rearrange(cx/fx, 'n -> n 1')  # (N,M)*(N,1)=(N,M)
        Ry = -Rz*xyz[..., 1]*rearrange(cy/fy, 'n -> n 1')  # (N,M)*(N,1)=(N,M)
        world = rearrange([Rx, Ry, Rz], 'c n m -> n m c', c=3) # (N,M,3)
    return world

def compute_world_visiblility(visibility, xyz, K, H, W, c2w):
    """
    Compute the visibility of xyz in NDC into a camera.
    Visibility = "inside camera"
    Inputs:
        visibility: (N) 0 or 1 valued visibilities, modified inplace
        xyz: (N, 3) world coordinates
        K: (3, 3) camera intrinsics
        H, W: image height and width
        c2w: (3, 4) camera to world matrix
    """
    c2w_ = torch.eye(4, device=xyz.device)
    c2w_[:3] = c2w
    w2c = torch.inverse(c2w_) # (4, 4)
    R, t = w2c[:3, :3], w2c[:3, 3:] # (3, 3) and (3, 1)
    xyz_cam = R @ xyz.T + t # (3, N)

    xyz_cam_in_front = xyz_cam[2] < 0 # (N) front is negative z axis!
    if xyz_cam_in_front.sum() == 0: # all points are behind the camera
        return

    xyz_cam = xyz_cam[:, xyz_cam_in_front] # (3, M)
    xyz_cam[1:] *= -1 # transform axes to "right down front"
    xyz_img = K @ xyz_cam # (3, M)
    xyz_img = xyz_img[:2] / xyz_img[2] # (2, M)

    visibility[xyz_cam_in_front] += \
        ((xyz_img[0]>=0)&(xyz_img[0]<W)&(xyz_img[1]>=0)&(xyz_img[1]<H)).float()