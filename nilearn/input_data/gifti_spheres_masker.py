"""
Transformer for computing seeds signals on the surface
------------------------------------------------------

Mask gifti images by spherical volumes for seed-region analyses
"""

from sklearn import neighbors

def _apply_surfmask_and_get_affinity(seeds, giimgs, mesh_coords, radius, allow_overlap,
                                     process_surfmask=None):
    process_surfmask_coords = mesh_coords[process_surfmask,:]
    clf = neighbors.NearestNeighbors(radius=radius)
    A = clf.fit(process_surfmask_coords).radius_neighbors_graph(seeds)
    del seeds
    A = A.tolil()

    print(process_surfmask.shape)
    print(giimgs.shape)

    X = giimgs[process_surfmask,:].T

    if not allow_overlap:
        if np.any(A.sum(axis=0) >= 2):
            raise ValueError('Overlap detected between spheres')

    return X, A



