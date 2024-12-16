import torch
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable
#from afl_modules.lidar_environment_sequence import population_code_estimate

def population_code_estimate(population_code: np.ndarray, param_dict: dict, estimate="max") -> np.ndarray:

    if isinstance(population_code, torch.Tensor):
        #population_code = population_code.detach().cpu().numpy()
        cuda_check = population_code.is_cuda
        if cuda_check:
            device = "cuda:"+str(population_code.get_device())
        else:
            device = 'cpu'
        #print("device: ", device)
        resolution_vector = torch.linspace(param_dict["val_min"], param_dict["val_max"], steps=param_dict["vector_length"]).to(device)
        #print(population_code.shape)
        #print(resolution_vector.unsqueeze(-1).shape)
        if estimate == "max":
            idx = population_code.argmax(axis=1)#.detach().cpu().numpy()
            max_value = resolution_vector[idx]
            return max_value
        elif estimate == "mean_pos": ## error for angle actions, doesn't give correct value
            mean_value = population_code @ resolution_vector.unsqueeze(-1) 
            #print(mean_value.shape)
            mean_value_normalized = mean_value / population_code.sum(axis=1).unsqueeze(-1)
            #print(mean_value_normalized.shape)
            return mean_value_normalized
        elif estimate == "mean_angle":
            mean = torch.atan2((torch.sin(resolution_vector)*population_code).sum(axis=-1), (torch.cos(resolution_vector)*population_code).sum(axis=-1))
            #print(mean.shape)
            return mean.unsqueeze(-1)
        else:
            raise ValueError("Estimate {} is not a valid choice!\nEither choose your estimate to be \"max\" or \"mean\".".format(estimate))
    else:
        resolution_vector = np.linspace(param_dict["val_min"], param_dict["val_max"], num=param_dict["vector_length"])
        if estimate == "max":
            idx = population_code.argmax(axis=1)#.detach().cpu().numpy()
            max_value = resolution_vector[idx]
            return max_value
        elif estimate == "mean_pos": ## error for angle actions, doesn't give correct value
            mean_value = np.dot(population_code, resolution_vector) 
            mean_value_normalized = mean_value / population_code.sum(axis=1)
            return mean_value_normalized
        elif estimate == "mean_angle":
            mean = np.arctan2((np.sin(resolution_vector)*population_code).sum(axis=-1), (np.cos(resolution_vector)*population_code).sum(axis=-1))
            return mean
        else:
            raise ValueError("Estimate {} is not a valid choice!\nEither choose your estimate to be \"max\" or \"mean\".".format(estimate))


class BeamDataVisualizer(object):
    def __init__(self):
        params_dict = {
            "type": "delta_pose",
            "size": 500000,
            "world_bounds": np.array([[1, 1], [10, 1], [10, 10], [1, 10]]),
            "obstacles_gen": {
                "obs_type": "rect",
                "num": 10,
                "width_mean": 1,
                "width_std": 1,
                "height_mean": 1,
                "height_std": 1,
                "seed": 1234,
            },
            "sample_gen": {
                "num_consecutive": 2,
                "pos_std": 0.1,
                "theta_std": np.deg2rad(10),
                "seed": 4321,
            },
            "sensorbeams": {"opening_angle": 180, "num_beams": 200}, # 200
        }

        self.world_bounds = np.array(params_dict['world_bounds'])
        self.seed_obs = params_dict['obstacles_gen']['seed']
        self.num_obs = params_dict['obstacles_gen']['num']
        w_mean, w_std, h_mean, h_std = params_dict['obstacles_gen']['width_mean'], params_dict['obstacles_gen'][
            'width_std'], params_dict['obstacles_gen']['height_mean'], params_dict['obstacles_gen']['height_std']
        self.obs_type = params_dict["obstacles_gen"]["obs_type"]
        self.obstacles = self.generate_random_rectangles(self.num_obs, w_mean, w_std, h_mean, h_std, self.world_bounds, self.seed_obs)

    def plot_world(self, seed, num_obs):
        self.ax_plot_world(self._ax, seed, num_obs)
        plt.show()

    def ax_plot_world(self, ax, name="NULL"):
        world_bounds = self.world_bounds
        # ax limits
        xlim_lower = np.min(world_bounds[:, 0]) - 1
        xlim_upper = np.max(world_bounds[:, 0]) + 1
        ylim_lower = np.min(world_bounds[:, 1]) - 1
        ylim_upper = np.max(world_bounds[:, 1]) + 1

        ax.set_xlim(xlim_lower, xlim_upper)
        ax.set_ylim(ylim_lower, ylim_upper)
        ax.set_aspect("equal")

        # plot world bounds
        wb_patch = Polygon(
            world_bounds, facecolor="None", edgecolor="black", linewidth=3, zorder=-1
        )
        ax.add_patch(wb_patch)

        # plot obstacles
        obstacles = self.obstacles
        obs_patch = generatePolygonPatchCollection(obstacles, "blue", 1.0)
        ax.add_collection(obs_patch)

        # set title
        #ax.set_title("num_obstacles={0}, seed={1}".format(self.num_obs, self.seed_obs))
        ax.set_title(name)

    def plot_pose(self, pos, theta):
        fig, ax = plt.subplots()
        self.ax_plot_pose(ax, pos, theta)
        plt.show()

    def visualize_sequence_connections_environment(self, action_position_parameters, position_parameters, angle_parameters, dataset, image_path, number_samples=100):
        fig, ax = plt.subplots(1,1,figsize=(5,5))
        i = 0
        for s1, s2, x1, y1, t1, x2, y2, t2, x12, y12, t12 in iter(dataset):
            if i >= number_samples:
                break

            pos = np.array([population_code_estimate(x1, position_parameters), population_code_estimate(y1, position_parameters)]).reshape((-1,))
            theta = population_code_estimate(t1, angle_parameters)
            points = self.convert_readings_to_cartesian(s1.numpy(), pos, theta)
            ax.plot(points[:,0], points[:,1], 'o', ms=1, color='y')

            # ax.plot(pos[0], pos[1], '.', ms=2, color='b')
            self.ax_plot_pose(ax, pos, theta, scale=0.15)

            pos2 = np.array([population_code_estimate(x2, position_parameters), population_code_estimate(y2, position_parameters)]).reshape((-1,))
            theta2 = population_code_estimate(t2, angle_parameters)
            self.ax_plot_pose(ax, pos2, theta2, scale=0.15, color='red')
            ax.arrow(pos[0], pos[1], pos2[0]-pos[0], pos2[1]-pos[1], color='g', lw=0.5, head_width=0.06)
            # This causes errors
            #dx = population_code_estimate(x12, action_position_parameters)
            #dy = population_code_estimate(y12, action_position_parameters)
            #ax.arrow(pos[0], pos[1], dx, dy, color='g', lw=0.5, head_width=0.06)
            i += 1

        #self.ax_plot_beam_samples(fig, ax1, X[indices_range], y[indices_range], np.random.randint(0, 4000, size=number_samples), losses, viridis, norm)
        fig.savefig(f'{image_path}.svg', format="svg")
        plt.show()

    def visualize_loss_environment(self, name, x, y, t, S, image_path, number_samples=1, loss_key="mse_loss", normalized=True, plot_beams=True, color='red', beam_alpha=0.5, connected=False):
        fig, ax1 = plt.subplots(1,1,figsize=(20,20))
        ax1.set_xlim([0, 10])
        ax1.set_ylim([0, 10])
        viridis = plt.cm.get_cmap('viridis', 256)
        norm = plt.Normalize(0, 40, clip=True)
        losses = 0

        if normalized:
            losses = norm(losses)

        green = np.array([0,1,0])
        red = np.array([1,0,0])
        alpha = np.linspace(1.0, 0.0, number_samples)

        def combine_colors(c1, c2,a):
            b = 1-a
            c3 = a*c1 + b*c2
            return tuple(c3)
        
        if number_samples > 1:
            #self.ax_plot_beam_samples(fig, ax1, x[0], y[0], t[0], S[0], 1, losses, viridis, norm, name=name, plot_beams=plot_beams, color=color)
            #self.ax_plot_world(ax1, name)
            for i in range(number_samples):
                color = combine_colors(green, red, alpha[i])
                # if i > 1:
                #     plot_beams = False
                # else:
                #     circle2 = plt.Circle((x[i], y[i]), 0.3, color='b', fill=False, linewidth=2)

                self.ax_plot_beam_samples(fig, ax1, x[i], y[i], t[i], S[i], number_samples-1, losses, viridis, norm, name=name, plot_beams=plot_beams, color=color, beam_alpha=beam_alpha)
                self.ax_plot_world(ax1, name)
                if (connected == True) and (i > 0):
                    ax1.arrow(x[i-1], y[i-1], x[i]-x[i-1], y[i]-y[i-1], color='y', lw=0.5, head_width=0.06)
                #ax1.add_patch(circle2)
        else:
            self.ax_plot_beam_samples(fig, ax1, x, y, t, S, number_samples, losses, viridis, norm, name=name, plot_beams=plot_beams)
            self.ax_plot_world(ax1, name)


        plt.show()
        return (fig, ax1)   #.savefig(f'{image_path}.svg', format="svg")
        

    def ax_plot_pose(self, ax, pos, theta, cm=None, loss=None, scale=0.25, color='red'):
        # green triangle in direction
        # generate triangle points
        angles = np.array([np.deg2rad(0), np.deg2rad(120), np.deg2rad(240)])
        points = np.array([np.cos(angles), np.sin(angles)])
        points[0] = points[0] * scale
        points[1:3] = points[1:3] * 0.5 * scale
        # transform triangle points to pose
        c = np.cos(theta)
        s = np.sin(theta)
        rotmat= np.array([[c, -s], [s, c]])

        points = np.dot(rotmat, points) + np.reshape(pos, (-1,1))
        points = points.transpose()

        #ax.plot(points[0,0], points[0,1], marker='o', color='black', zorder=10.1, ms=10*scale)
        ax.arrow(pos[0], pos[1], points[0,0]-pos[0], points[0,1]-pos[1], color='g', lw=0.5, head_width=0.06)

        if cm is None or loss is None:
            ax.plot(pos[0], pos[1], marker='x', color=color, zorder=10.1, ms=20*scale)
        else:
            ax.plot(pos[0], pos[1], marker='x', color=cm(loss), zorder=10.1, ms=20*scale)

    def ax_plot_single_position_samples(self, ax, y, sample_indices, first_pose=True):
        if first_pose:
            pose_index = 1 # row in the sample
            c = 'b'
        else:
            pose_index = 2 # row in the sample
            c = 'g'
        for ind in sample_indices:
            ax.plot(y[sample_indices,pose_index,0], y[sample_indices,pose_index,1], '.', ms=2, color=c)
        ax.set_title('num_position_samples={0}'.format(len(sample_indices)))

    def plot_angle_distribution(self, y, sample_indices, n_bins=18):
        fig, (ax0, ax1) = plt.subplots(1,2)
        self.ax_plot_angle_distribution(ax0, y, sample_indices, first_pose=True, n_bins=18)
        self.ax_plot_angle_distribution(ax1, y, sample_indices, first_pose=False, n_bins=18)
        plt.show()

    def ax_plot_angle_distribution(self, ax, y, sample_indices, first_pose=True, n_bins=18):
        if first_pose:
            pose_index = 1 # row in the sample
            c = 'b'
        else:
            pose_index = 2 # row in the sample
            c = 'g'
        for ind in sample_indices:
            ax.hist(np.rad2deg(y[sample_indices,pose_index,2]), bins=n_bins,color=c)
        ax.set_title('num_angles={0}'.format(len(sample_indices)))

    def plot_beam_samples(self, X, y, sample_indices, first_pose=True):
        fig, ax = plt.subplots()
        self.ax_plot_beam_samples(ax, X, y, sample_indices, first_pose)
        plt.show()

    def ax_plot_beam_samples(self, fig, ax, X, Y, T, S, sample_indices, losses, colormap, norm, plot_beams, color, beam_alpha, first_pose=True, name="NULL"):

        readings = S
        x = X
        y = Y
        theta = T
        pos = np.array([x, y])

        points = self.convert_readings_to_cartesian(readings, pos, theta)
        if plot_beams == True:
            ax.plot(points[:, 0], points[:, 1], 'o', ms=7, color=color, alpha=beam_alpha)
        # ax.plot(pos[0], pos[1], '.', ms=2, color='b')
        self.ax_plot_pose(ax, pos, theta, colormap, color=color, scale=0.5)

        # colorbar
        # divider = make_axes_locatable(ax)
        # cax = divider.append_axes('right', size='5%', pad=0.05)
        # fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=colormap), cax=cax, orientation='vertical')
        #ax.set_title('num_pose_samples={0}'.format(len(sample_indices)))
        ax.set_title(name)

    def convert_readings_to_cartesian(self, readings, pos, theta, angles=None):
        if angles is None:
            opening_angle = 180
            num_beams = 200
            beam_dirs, beam_angles = self.generate_beam_dir_vecs(
                opening_angle, num_beams, direction_angle=0)
            angles = beam_angles

        rot_angles = angles+theta
        dir_vecs = np.transpose(np.array([np.cos(rot_angles), np.sin(rot_angles)]))
        points = dir_vecs * readings.reshape(-1,1) + pos
        return points

    def generate_beam_dir_vecs(self, opening_angle, num_beams, direction_angle=0, visible_range=1.0):
        # angles of the circle approximation
        angles = np.linspace(direction_angle - opening_angle / 2, direction_angle + opening_angle / 2, num_beams)

        # first approx.
        rad = np.deg2rad(angles[0])
        l = [np.cos(rad) * visible_range, np.sin(rad) * visible_range]

        dirs = np.zeros(shape=(num_beams,2))

        i = 0
        # add vectors to list
        for a in angles:
            rotateMatrix = self.rotationMatrix(np.deg2rad(a) - rad)
            tempPoint = np.dot(rotateMatrix, l)
            # print "angle : ", a, " point : ", tempPoint, c, s
            dirs[i,:] = tempPoint
            i+=1

        return dirs, np.deg2rad(angles)

    def rotationMatrix(self, radian):
        c = np.cos(radian)
        s = np.sin(radian)
        return np.array([[c, -s], [s, c]])

    def generate_random_rectangles(
        self, num, width_mean, width_std, height_mean, height_std, world_bounds, seed=None
    ):
        from numpy.random import default_rng

        rng = default_rng(seed)
        xlim_lower = np.min(world_bounds[:, 0])
        xlim_upper = np.max(world_bounds[:, 0])
        ylim_lower = np.min(world_bounds[:, 1])
        ylim_upper = np.max(world_bounds[:, 1])

        rects = []

        for i in range(num):
            # sample center pos
            x_pos = rng.uniform(xlim_lower, xlim_upper)
            y_pos = rng.uniform(ylim_lower, ylim_upper)

            # sample width + height
            w = rng.normal(width_mean, width_std)
            h = rng.normal(height_mean, height_std)

            ll = np.array([x_pos - w / 2, y_pos - h / 2])
            lr = np.array([x_pos + w / 2, y_pos - h / 2])
            ur = np.array([x_pos + w / 2, y_pos + h / 2])
            ul = np.array([x_pos - w / 2, y_pos + h / 2])

            rect = np.array([ll, lr, ur, ul])
            rects.append(rect)

        return rects

        
def generatePolygonPatchCollection(listOfNumpyPolygons, colorV="blue", alphaV=0.4):
    polygons = []
    for p in listOfNumpyPolygons:
        polygons.append(Polygon(p, closed=True))

    return PatchCollection(polygons, alpha=alphaV, color=colorV)

def load_deltapose_dataset(path_to_dataset_folder, dataset_name):
    from pathlib import Path
    import json
    import pickle

    dataset_path = Path(path_to_dataset_folder) / dataset_name

    # load params
    params_path = dataset_path / 'params.json'
    with params_path.open() as f:
        params = json.load(f)

    # load dataset 
    datasetfile_path = dataset_path / str(dataset_name+'.pickle')
    with datasetfile_path.open(mode='rb') as f:
        dataset = pickle.load(f)
    X, y, X_col_angles = dataset['X'], dataset['y'], dataset['X_columns']

    return X, y, X_col_angles, params