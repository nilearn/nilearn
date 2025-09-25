import pytest

from nilearn.surface.clusters import (
    compute_adjacency_matrix,
    find_surface_clusters,
)

# TODO check that clusters with same number of faces do not get same label


def test_compute_adjacency_matrix(surf_mesh):
    adjacency_matrix = compute_adjacency_matrix(surf_mesh.parts["left"])
    assert adjacency_matrix.shape == (4, 4)
    assert adjacency_matrix.size == 12

    adjacency_matrix = compute_adjacency_matrix(surf_mesh.parts["right"])
    assert adjacency_matrix.shape == (5, 5)
    assert adjacency_matrix.size == 18


@pytest.mark.parametrize(
    "mask, expected_n_clusters",
    [
        ([False, False, False, False], 0),
        ([True, False, False, False], 1),
        ([True, True, True, True], 1),
    ],
)
def test_find_surface_clusters_4_faces(mask, expected_n_clusters, surf_mesh):
    clusters, _ = find_surface_clusters(surf_mesh.parts["left"], mask)
    assert len(clusters) == expected_n_clusters


@pytest.mark.parametrize(
    "mask, expected_n_clusters",
    [
        ([False, False, False, False, False], 0),
        ([True, False, False, False, False], 1),
        ([False, True, False, True, False], 1),
        ([True, True, True, True, True], 1),
    ],
)
def test_find_surface_clusters_5_faces(mask, expected_n_clusters, surf_mesh):
    clusters, _ = find_surface_clusters(surf_mesh.parts["right"], mask)
    assert len(clusters) == expected_n_clusters
