import numpy as np
import torch
from .lidar_visualizer_patrick import BeamDataVisualizer


class BaseAFLEnv():
    def __init__(self, starting_pose: np.ndarray):
        self.starting_pose = starting_pose
        self.current_pose = starting_pose

    def _get_line_segments_from_rectangles(self, rectangles: np.ndarray) -> np.ndarray:
        line_segment_list = []
        for rectangle in rectangles:
            line_1 = np.empty((1,2,2))
            line_2 = np.empty((1,2,2))
            line_3 = np.empty((1,2,2))
            line_4 = np.empty((1,2,2))

            line_1[:,0,0] = rectangle[0,0]
            line_1[:,0,1] = rectangle[0,1]
            line_1[:,1,0] = rectangle[1,0]
            line_1[:,1,1] = rectangle[0,1]

            line_2[:,0,0] = rectangle[1,0]
            line_2[:,0,1] = rectangle[0,1]
            line_2[:,1,0] = rectangle[1,0]
            line_2[:,1,1] = rectangle[1,1]
            
            line_3[:,0,0] = rectangle[1,0]
            line_3[:,0,1] = rectangle[1,1]
            line_3[:,1,0] = rectangle[0,0]
            line_3[:,1,1] = rectangle[1,1]

            line_4[:,0,0] = rectangle[0,0]
            line_4[:,0,1] = rectangle[1,1]
            line_4[:,1,0] = rectangle[0,0]
            line_4[:,1,1] = rectangle[0,1]

            line_segment_list.append(line_1)
            line_segment_list.append(line_2)
            line_segment_list.append(line_3)
            line_segment_list.append(line_4)
    
        return np.concatenate(line_segment_list, axis=0)
    
    def _check_collision(self, rays, line_segments, pos):  # https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection
        x1 = line_segments[:,0,0]
        y1 = line_segments[:,0,1]
        x2 = line_segments[:,1,0]
        y2 = line_segments[:,1,1]

        x3 = rays[:,0,0]
        y3 = rays[:,0,1]
        x4 = rays[:,1,0]
        y4 = rays[:,1,1]

        a = np.subtract.outer(x1,x3)    # (2, 20)
        b = np.subtract.outer(y1,y3)    # (2, 20)

        numerator_t = a*(y3-y4) - b*(x3-x4)
        numerator_u = a*(y1-y2)[:,None] - b*(x1-x2)[:,None]

        denominator = np.multiply.outer((x1-x2), (y3-y4)) - np.multiply.outer((y1-y2), (x3-x4))

        if np.any(denominator, where=0):
            print("denominator is zero!")
            return None

        t = numerator_t / denominator   # rays
        u = numerator_u / denominator   # obstacles

        conditions = (0<=t) * (t<=1) * (0<=u)# * (u<=1)  # boolean table of intersections between all objects and all rays
        #conditions = (0<=u) * (u<=1)  # boolean table of intersections between all objects and all hypothetical rays (direction of rays)
        condition = np.logical_or.reduce(conditions)    # boolean vector of rays having any intersection point with an obstacle

        if condition.all() == True:
            x_intersect_list = []
            y_intersect_list = []
            for i in range(rays.shape[0]):
                ray_intersects_x = x3[i] + u[:,i][conditions[:,i]]*(x4[i]-x3[i])
                ray_intersects_y = y3[i] + u[:,i][conditions[:,i]]*(y4[i]-y3[i])

                distance = np.sqrt((ray_intersects_x-pos[0])**2 + (ray_intersects_y-pos[1])**2)
                closest_intersect_idx = np.argmin(distance)

                x_intersect_list.append(ray_intersects_x[closest_intersect_idx])
                y_intersect_list.append(ray_intersects_y[closest_intersect_idx])

            x_intersect = np.array(x_intersect_list)
            y_intersect = np.array(y_intersect_list)

            intersection_points = np.stack([x_intersect, y_intersect], axis=1)
        else:
            return (None, condition)

        return (intersection_points, condition)
    
    def _calc_offset(self, pos: np.ndarray, angles: np.ndarray, magnitudes: np.ndarray) -> np.ndarray:
        offset_x = np.cos(angles)*magnitudes
        offset_y = np.sin(angles)*magnitudes
        offset = pos + np.stack([offset_x, offset_y], axis=1)
        return offset
    

    def _get_beam_reading_os(self, pose: np.ndarray, line_segments: np.ndarray, num_rays: int, fov: int) -> tuple:
        pos = np.array([pose[:2]]).repeat(num_rays, 0)
        angles = np.linspace(-(fov/2)+pose[2],(fov/2)+pose[2], num=num_rays)
        
        magnitudes = np.ones(num_rays)*0.01

        offset_x = np.cos(angles)*magnitudes
        offset_y = np.sin(angles)*magnitudes

        offset = np.stack([offset_x, offset_y], axis=1)

        offset_array = pos + offset

        rays = np.stack([pos, offset_array], axis=1)

        intersection_points, mask = self._check_collision(
            rays=rays, 
            line_segments=line_segments,
            pos=pos[0]
        )

        if mask.all() == True:
            magnitudes = np.sqrt(((intersection_points-pos)**2).sum(-1))
            offset_array = self._calc_offset(pos, angles, magnitudes)
            rays = np.stack([pos, offset_array], axis=1)
            pass
        else:
            raise ValueError("not all rays hit a target!")
            
        return (intersection_points, rays, magnitudes)

    def reset(self, num_rays: torch.tensor, fov: torch.tensor):
        self.current_pose = self.starting_pose

        intersection_points, rays, magnitudes = self._get_beam_reading_os(
            pose=self.current_pose,
            line_segments=self.line_segments,
            num_rays=num_rays,
            fov=np.deg2rad(fov)
        )

        return (intersection_points, rays, magnitudes, self.current_pose)
    
    def plot_env(self, text, samples, data_dict, plot_beams, beam_alpha):
        image_path = ""

        bdv_gt = BeamDataVisualizer()
        fig_env_pred = bdv_gt.visualize_loss_environment(
            text, 
            data_dict['x'], 
            data_dict['y'], 
            data_dict['t'], 
            data_dict['si'], 
            image_path, 
            samples, 
            "mse_loss", 
            True,
            plot_beams=plot_beams,
            beam_alpha=beam_alpha,
            connected=True
        )