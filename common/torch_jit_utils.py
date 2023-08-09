import torch
import torch.nn.functional as F
import numpy as np

@torch.jit.script
def copysign(a, b):
    # type: (float, Tensor) -> Tensor
    a = torch.tensor(a, device=b.device, dtype=torch.float).repeat(b.shape[0])
    return torch.abs(a) * torch.sign(b)

@torch.jit.script
def get_euler_xyz(q):
    qx, qy, qz, qw = 0, 1, 2, 3
    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (q[:, qw] * q[:, qx] + q[:, qy] * q[:, qz])
    cosr_cosp = q[:, qw] * q[:, qw] - q[:, qx] * \
        q[:, qx] - q[:, qy] * q[:, qy] + q[:, qz] * q[:, qz]
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2.0 * (q[:, qw] * q[:, qy] - q[:, qz] * q[:, qx])
    pitch = torch.where(torch.abs(sinp) >= 1, copysign(
        np.pi / 2.0, sinp), torch.asin(sinp))

    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (q[:, qw] * q[:, qz] + q[:, qx] * q[:, qy])
    cosy_cosp = q[:, qw] * q[:, qw] + q[:, qx] * \
        q[:, qx] - q[:, qy] * q[:, qy] - q[:, qz] * q[:, qz]
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return roll % (2*np.pi), pitch % (2*np.pi), yaw % (2*np.pi)

@torch.jit.script 
def localToGlobalRot(roll, pitch, yaw, x, y, z):
    cosa = torch.cos(yaw)
    sina = torch.sin(yaw)

    cosb = torch.cos(pitch)
    sinb = torch.sin(pitch)

    cosy = torch.cos(roll)
    siny = torch.sin(roll)

    xp = x*cosa*cosb + y*(cosa*sinb*siny - sina*cosy) + z*(cosa*sinb*cosy + sina*siny)
    yp = x*sina*cosb + y*(sina*sinb*siny + cosa*cosy) + z*(sina*sinb*cosy - cosa*siny)
    zp = -x*sinb + y*cosb*siny + z*cosb*cosy 

    return xp, yp, zp

@torch.jit.script 
def globalToLocalRot(roll, pitch, yaw, x, y, z):
    cosa = torch.cos(yaw)
    sina = torch.sin(yaw)

    cosb = torch.cos(pitch)
    sinb = torch.sin(pitch)

    cosy = torch.cos(roll)
    siny = torch.sin(roll)

    xp = x*cosa*cosb                    + y*sina*cosb                    - z*sinb
    yp = x*(cosa*sinb*siny - sina*cosy) + y*(sina*sinb*siny + cosa*cosy) + z*cosb*siny
    zp = x*(cosa*sinb*cosy + sina*siny) + y*(sina*sinb*cosy - cosa*siny) + z*cosb*cosy

    return xp, yp, zp

@torch.jit.script
def quat_from_euler_xyz(roll, pitch, yaw):
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)

    qw = cy * cr * cp + sy * sr * sp
    qx = cy * sr * cp - sy * cr * sp
    qy = cy * cr * sp + sy * sr * cp
    qz = sy * cr * cp - cy * sr * sp

    return torch.stack([qx, qy, qz, qw], dim=-1)