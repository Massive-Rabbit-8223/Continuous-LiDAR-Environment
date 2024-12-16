import numpy as np
import torch
from .base_env import BaseAFLEnv

# TODO: Create Parent class containing all the basic mechanics like checking collision and computing beamreading
# TODO: Get initial state of the world (pose, beamreading) and also make use of it in the reset function

class SimpleAFLEnv(BaseAFLEnv):
    """
    Environment used in the Action Field Learning Master Thesis by Patrick Mederitsch
    https://epub.jku.at/download/pdf/9098470.pdf
    This environment consist of 10 rectangular obstacles
    """
    def __init__(self, starting_pose: torch.tensor):
        self.starting_pose = starting_pose

        super().__init__(self.starting_pose)
        
        world_bounds = np.array([[
            [1, 1], 
            [10, 10]
        ]])

        obstacle_bounds = np.array([
            # Obstacle 1
            [[ 8.91985225,  3.84545202],
            [10.66074355,  4.99807121]],
            # Obstacle 2
            [[3.63246185, 1.09008461],
            [4.11128521, 3.03555758]],
            # Obstacle 3
            [[9.43293506, 2.21096876],
            [9.92049135, 4.53472772]],
            # Obsatcle 4
            [[6.94135996, 6.35929963],
            [7.20650368, 7.51843864]],
            # Obstacle 5
            [[1.78852858, 8.83265186],
            [3.30866274, 8.83481765]],
            # Obstacle 6
            [[5.94550616, 6.42045643],
            [8.13677819, 6.57786723]],
            # Obstacle 7
            [[4.46677218, 5.2867644 ],
            [5.4343571 , 6.29994598]],
            # Obstacle 8
            [[7.56080683, 4.1982659 ],
            [9.89202105, 5.45710442]],
            # Obstacle 9
            [[ 2.31954943,  9.68308763],
            [ 2.44298565, 10.1775785 ]],
            # Obstacle 10
            [[0.86895121, 4.83750394],
            [2.69494379, 5.59028893]]
        ])

        total_obstacles = np.concatenate(
            [world_bounds, obstacle_bounds], 
            axis=0
        )

        self.line_segments = self._get_line_segments_from_rectangles(total_obstacles)  # convert obstacles to line segments

    def step(self, action: torch.tensor, num_rays: torch.tensor, fov: torch.tensor) -> tuple:

        self.next_pose = self._nh_apply_action(self.current_pose, action)
        
        intersection_points, rays, magnitudes = self._get_beam_reading_os(
            pose=self.next_pose,
            line_segments=self.line_segments,
            num_rays=num_rays,
            fov=np.deg2rad(fov)
        )

        self.current_pose = self.next_pose

        return (intersection_points, rays, magnitudes, self.current_pose)
    
    def _nh_apply_action(self, pose_1: torch.tensor, action: torch.tensor) -> torch.tensor:
        ## TODO: Doc-string, comment ##
        pose_2 = torch.zeros_like(pose_1)

        x_1 = pose_1[0]
        y_1 = pose_1[1]
        theta_1 = pose_1[2]

        relative_x_offset = action[0]
        relative_y_offset = action[1]
        delta_theta = action [2]

        offset_distance = torch.sqrt(torch.pow(relative_x_offset, 2) + torch.pow(relative_y_offset, 2))
        if offset_distance == 0:
            beta = 0
        else:
            beta = torch.asin(relative_y_offset / offset_distance)  # changed from acos to asin 18.11.2022
        alpha = theta_1 + beta  # changed from - beta to + beta 03.12.2022

        delta_x = torch.cos(alpha) * offset_distance
        delta_y = torch.sin(alpha) * offset_distance

        x_2 = x_1 + delta_x
        y_2 = y_1 + delta_y
        theta_2 = theta_1 + delta_theta

        if theta_2 > (2*torch.pi):
            theta_2 = theta_2 - (2*torch.pi)
        elif theta_2 < 0:
            theta_2 = (2*torch.pi) + theta_2
        else:
            theta_2 = theta_2

        pose_2[0] = x_2
        pose_2[1] = y_2
        pose_2[2] = theta_2

        return pose_2
