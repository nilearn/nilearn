import numpy as np
from matplotlib import pyplot as plt
from nilearn.plotting import surf_plotting


def _piecewise_const_image():
    w, h = 7, 10
    image = np.random.uniform(0, 1, size=(w, h))
    image -= image.min()
    image /= image.max()
    big_image = np.empty((10 * w, 10 * h))
    for i in range(w):
        for j in range(h):
            big_image[10 * i:10 * i + 10, 10 * j:10 * j + 10] = image[i, j]
    return big_image


def _random_mesh(image_shape, n_nodes=5):
    x = np.random.uniform(0, image_shape[0], size=(n_nodes, ))
    y = np.random.uniform(0, image_shape[1], size=(n_nodes, ))
    return np.asarray([x, y]).T


def show_sampling(ball_radius=10, n_nodes=5, n_points=7, link_points=True):
    image = _piecewise_const_image()
    mesh = _random_mesh(image.shape, n_nodes=n_nodes)
    values, sample_points = surf_plotting._vol_to_surf_ball_sampling(
        image, mesh, np.eye(3), n_points=n_points, ball_radius=ball_radius)
    fig, ax = plt.subplots()
    ax.imshow(image, cmap='gray', vmin=0, vmax=1)
    ax.scatter(mesh[:, 1], mesh[:, 0], c=values, s=300, cmap='gray',
               edgecolors='r', vmin=0, vmax=1, zorder=2)
    for sp, mp in zip(sample_points, mesh):
        ax.scatter(sp[:, 1], sp[:, 0], marker='x', color='blue', zorder=3)
        if link_points:
            for s in sp:
                ax.plot(
                    [mp[1], s[1]], [mp[0], s[0]],
                    color='red',
                    alpha=.5,
                    zorder=1)
    return ax


if __name__ == '__main__':
    ax = show_sampling()
    plt.show()
