import sys
import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *

############################
# Utility Functions
############################

def rotate_and_resample(voxel, R, translation, upsample_factor=4):
    """
    Rotate and translate a voxel grid using some interpolation.
    Placeholder: Implement a trilinear interpolation or a known 3D warp function.
    voxel: (D,H,W) binary or float tensor.
    R: 3x3 rotation matrix (torch.tensor)
    translation: 3-vector (torch.tensor)
    upsample_factor: int
    returns: transformed voxel grid at original resolution after intermediate upsampling.

    Steps (conceptual):
    1. Upsample voxel by factor (D*uf, H*uf, W*uf) using interpolation.
    2. Apply rotation and translation to sampling coordinates.
    3. Resample down or keep at high resolution and then integrate overlap.
    """
    # For demonstration, just return voxel as is
    return voxel.clone()

def compute_voxel_overlap(voxA, voxB):
    # Assuming both are aligned and same shape
    return (voxA * voxB).sum().item()

def parametric_backtrack(initial_pose, linear_vel, angular_vel, steps=10, dt=0.01):
    """
    Move backward in time (negative dt) from initial_pose.
    initial_pose: (R,t) rotation (3x3), translation (3,)
    linear_vel: (3,)
    angular_vel: (3,) representing angular velocity in some axis-angle form
    steps: int, how many steps backward
    dt: time step
    returns a list of (R,t) poses.
    """
    # Simplified: just step linearly and rotate slightly per step
    poses = []
    R, t = initial_pose
    for i in range(steps):
        # Reverse linear vel
        t = t - linear_vel * dt
        # Reverse angular vel (approx): rotate by -angular_vel * dt
        # Convert angular_vel to a rotation matrix (small-angle approximation)
        angle = torch.norm(angular_vel)
        if angle > 1e-9:
            axis = angular_vel / angle
            theta = -angle * dt
            Rdelta = small_rotation_matrix(axis, theta)
            R = Rdelta @ R
        poses.append((R.clone(), t.clone()))
    return poses

def small_rotation_matrix(axis, theta):
    # Rodrigues formula for small rotation
    ux, uy, uz = axis
    cost = math.cos(theta)
    sint = math.sin(theta)
    R = torch.tensor([
        [cost+ux*ux*(1-cost), ux*uy*(1-cost)-uz*sint, ux*uz*(1-cost)+uy*sint],
        [uy*ux*(1-cost)+uz*sint, cost+uy*uy*(1-cost), uy*uz*(1-cost)-ux*sint],
        [uz*ux*(1-cost)-uy*sint, uz*uy*(1-cost)+ux*sint, cost+uz*uz*(1-cost)]
    ], dtype=torch.float32)
    return R

############################
# Neural Network
############################

class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(3+3+1,64), # linear_vel(3), angular_vel(3), and overlap integral(1)
            nn.ReLU(),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,1) # predict force or category
        )
    def forward(self, vel):
        return self.fc(vel)

############################
# Main Class
############################

class CollisionAnalysis:
    def __init__(self, voxelA, voxelB, RA, tA, RB, tB, device='cpu'):
        """
        voxelA, voxelB: high-resolution 3D volumes (torch tensors)
        RA, tA: rotation (3x3), translation(3,) of shape A
        RB, tB: rotation (3x3), translation(3,) of shape B
        """
        self.voxelA = voxelA.to(device)
        self.voxelB = voxelB.to(device)
        self.RA = RA.to(device)
        self.tA = tA.to(device)
        self.RB = RB.to(device)
        self.tB = tB.to(device)
        self.device = device

        # A small NN to train
        self.net = SimpleMLP().to(device)
        self.opt = optim.Adam(self.net.parameters(), lr=1e-3)
        self.loss_fn = nn.MSELoss()

    def compute_overlap_integral(self, RA, tA, RB, tB, upsample_factor=4):
        """
        Rotate and translate voxelB to voxelA's frame, do interpolation
        and compute overlap integral (a scalar representing how deeply they intersect).
        """
        # Assume voxelA is reference frame
        # Transform voxelB
        voxelB_trans = rotate_and_resample(self.voxelB, RB, tB, upsample_factor)
        voxelA_trans = rotate_and_resample(self.voxelA, RA, tA, upsample_factor)

        overlap = compute_voxel_overlap(voxelA_trans, voxelB_trans)
        return overlap

    def find_parametric_contact(self, RA, tA, RB, tB, linear_vel, angular_vel, steps=10, dt=0.01):
        """
        Move backward in time to find when shapes first come into certain contact condition.
        Just a placeholder method:
        We do a parametric backtrack and pick the pose with minimal overlap or a condition.
        """
        poses = parametric_backtrack((RB, tB), linear_vel, angular_vel, steps=steps, dt=dt)
        # Evaluate overlap at each pose, pick the best condition
        best_pose = (RB,tB)
        min_overlap = float('inf')
        for (Rp, tp) in poses:
            overlap = self.compute_overlap_integral(RA, tA, Rp, tp)
            if overlap < min_overlap:
                min_overlap = overlap
                best_pose = (Rp, tp)
        return best_pose, min_overlap

    def train_on_scenarios(self, scenarios, epochs=5, batch_size=8):
        """
        scenarios: list of (linear_vel, angular_vel, RA,tA,RB,tB)
        We compute overlap integral at final and found parametric contact point
        and train NN to predict that from the velocities + overlap integral
        """
        data = []
        for (lin_vel, ang_vel, RA, tA, RB, tB) in scenarios:
            # Find parametric contact scenario
            (Rp, tp), overlap = self.find_parametric_contact(RA,tA,RB,tB, lin_vel, ang_vel)
            # Input: lin_vel(3), ang_vel(3), overlap(1)
            inp = torch.cat([lin_vel, ang_vel, torch.tensor([overlap],dtype=torch.float32,device=self.device)], dim=0)
            # Target: For demonstration, let's say target = overlap * 0.1 (placeholder)
            target = torch.tensor([overlap*0.1], dtype=torch.float32, device=self.device)
            data.append((inp,target))

        # Train
        self.net.train()
        N = len(data)
        for epoch in range(epochs):
            random.shuffle(data)
            total_loss = 0
            for i in range(0,N,batch_size):
                batch = data[i:i+batch_size]
                inp = torch.stack([x[0] for x in batch])
                tgt = torch.stack([x[1] for x in batch])
                self.opt.zero_grad()
                pred = self.net(inp)
                loss = self.loss_fn(pred, tgt)
                loss.backward()
                self.opt.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}/{epochs}, Loss={total_loss/(N/batch_size):.4f}")

############################
# Pygame Setup (No Glut)
############################

def init_pygame():
    pygame.init()
    screen = pygame.display.set_mode((800,600), DOUBLEBUF|OPENGL)
    pygame.display.set_caption("Collision Analysis Demo")
    glViewport(0,0,800,600)
    glClearColor(0,0,0,1)
    return screen

def handle_user_input():
    changes = {}
    for event in pygame.event.get():
        if event.type == QUIT:
            sys.exit(0)
        elif event.type == KEYDOWN:
            if event.key == K_ESCAPE:
                sys.exit(0)
            elif event.key == ord('r'):
                changes['resolution'] = 'increase'
            elif event.key == ord('d'):
                changes['delay'] = 'increase'
    return changes

def render_demo():
    glClear(GL_COLOR_BUFFER_BIT)
    # Just clearing screen in this demo
    pygame.display.flip()

############################
# Main Demo
############################

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Suppose voxelA and voxelB are high-res tensors
    # Here we use small dummy volumes
    voxelA = (torch.rand(32,32,32)>0.95).float()
    voxelB = (torch.rand(32,32,32)>0.95).float()
    RA = torch.eye(3)
    tA = torch.zeros(3)
    RB = torch.eye(3)
    tB = torch.zeros(3)

    analysis = CollisionAnalysis(voxelA, voxelB, RA, tA, RB, tB, device=device)

    # Create some scenarios
    scenarios = []
    for _ in range(20):
        lin_vel = torch.randn(3)
        ang_vel = torch.randn(3)*0.1
        # random transforms
        RA_s = torch.eye(3)
        tA_s = torch.zeros(3)
        RB_s = torch.eye(3)
        tB_s = torch.zeros(3)
        scenarios.append((lin_vel.to(device),ang_vel.to(device),RA_s.to(device),tA_s.to(device),RB_s.to(device),tB_s.to(device)))

    analysis.train_on_scenarios(scenarios, epochs=2)

    screen = init_pygame()
    running = True
    while running:
        changes = handle_user_input()
        render_demo()
        pygame.time.wait(100)

if __name__ == "__main__":
    main()
