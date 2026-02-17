from math import prod
from pathlib import Path

import numpy as np
import pytest

from nibabel import pointset as ps
from nibabel.affines import apply_affine
from nibabel.fileslice import strided_scalar
from nibabel.optpkg import optional_package
from nibabel.spatialimages import SpatialImage
from nibabel.tests.nibabel_data import get_nibabel_data

h5, has_h5py, _ = optional_package('h5py')

FS_DATA = Path(get_nibabel_data()) / 'nitest-freesurfer'


class TestPointsets:
    rng = np.random.default_rng()

    @pytest.mark.parametrize('shape', [(5, 2), (5, 3), (5, 4)])
    @pytest.mark.parametrize('homogeneous', [True, False])
    def test_init(self, shape, homogeneous):
        coords = self.rng.random(shape)

        if homogeneous:
            coords = np.column_stack([coords, np.ones(shape[0])])

        points = ps.Pointset(coords, homogeneous=homogeneous)
        assert np.allclose(points.affine, np.eye(shape[1] + 1))
        assert points.homogeneous is homogeneous
        assert (points.n_coords, points.dim) == shape

        points = ps.Pointset(coords, affine=np.diag([2] * shape[1] + [1]), homogeneous=homogeneous)
        assert np.allclose(points.affine, np.diag([2] * shape[1] + [1]))
        assert points.homogeneous is homogeneous
        assert (points.n_coords, points.dim) == shape

        # Badly shaped affine
        with pytest.raises(ValueError):
            ps.Pointset(coords, affine=[0, 1])

        # Badly valued affine
        with pytest.raises(ValueError):
            ps.Pointset(coords, affine=np.ones((shape[1] + 1, shape[1] + 1)))

    @pytest.mark.parametrize('shape', [(5, 2), (5, 3), (5, 4)])
    @pytest.mark.parametrize('homogeneous', [True, False])
    def test_affines(self, shape, homogeneous):
        orig_coords = coords = self.rng.random(shape)

        if homogeneous:
            coords = np.column_stack([coords, np.ones(shape[0])])

        points = ps.Pointset(coords, homogeneous=homogeneous)
        assert np.allclose(points.get_coords(), orig_coords)

        # Apply affines
        scaler = np.diag([2] * shape[1] + [1])
        scaled = scaler @ points
        assert np.array_equal(scaled.coordinates, points.coordinates)
        assert np.array_equal(scaled.affine, scaler)
        assert np.allclose(scaled.get_coords(), 2 * orig_coords)

        flipper = np.eye(shape[1] + 1)
        # [[1, 0, 0], [0, 1, 0], [0, 0, 1]] becomes [[0, 1, 0], [1, 0, 0], [0, 0, 1]]
        flipper[:-1] = flipper[-2::-1]
        flipped = flipper @ points
        assert np.array_equal(flipped.coordinates, points.coordinates)
        assert np.array_equal(flipped.affine, flipper)
        assert np.allclose(flipped.get_coords(), orig_coords[:, ::-1])

        # Concatenate affines, with any associativity
        for doubledup in [(scaler @ flipper) @ points, scaler @ (flipper @ points)]:
            assert np.array_equal(doubledup.coordinates, points.coordinates)
            assert np.allclose(doubledup.affine, scaler @ flipper)
            assert np.allclose(doubledup.get_coords(), 2 * orig_coords[:, ::-1])

    def test_homogeneous_coordinates(self):
        ccoords = self.rng.random((5, 3))
        hcoords = np.column_stack([ccoords, np.ones(5)])

        cartesian = ps.Pointset(ccoords)
        homogeneous = ps.Pointset(hcoords, homogeneous=True)

        for points in (cartesian, homogeneous):
            assert np.array_equal(points.get_coords(), ccoords)
            assert np.array_equal(points.get_coords(as_homogeneous=True), hcoords)

        affine = np.diag([2, 3, 4, 1])
        cart2 = affine @ cartesian
        homo2 = affine @ homogeneous

        exp_c = apply_affine(affine, ccoords)
        exp_h = (affine @ hcoords.T).T
        for points in (cart2, homo2):
            assert np.array_equal(points.get_coords(), exp_c)
            assert np.array_equal(points.get_coords(as_homogeneous=True), exp_h)


def test_GridIndices():
    # 2D case
    shape = (2, 3)
    gi = ps.GridIndices(shape)

    assert gi.dtype == np.dtype('u1')
    assert gi.shape == (6, 2)
    assert repr(gi) == '<GridIndices(2, 3)>'

    gi_arr = np.asanyarray(gi)
    assert gi_arr.dtype == np.dtype('u1')
    assert gi_arr.shape == (6, 2)
    # Tractable to write out
    assert np.array_equal(gi_arr, [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]])

    shape = (2, 3, 4)
    gi = ps.GridIndices(shape)

    assert gi.dtype == np.dtype('u1')
    assert gi.shape == (24, 3)
    assert repr(gi) == '<GridIndices(2, 3, 4)>'

    gi_arr = np.asanyarray(gi)
    assert gi_arr.dtype == np.dtype('u1')
    assert gi_arr.shape == (24, 3)
    # Separate implementation
    assert np.array_equal(gi_arr, np.mgrid[:2, :3, :4].reshape(3, -1).T)


class TestGrids(TestPointsets):
    @pytest.mark.parametrize('shape', [(5, 5, 5), (5, 5, 5, 5), (5, 5, 5, 5, 5)])
    def test_from_image(self, shape):
        # Check image is generates voxel coordinates
        affine = np.diag([2, 3, 4, 1])
        img = SpatialImage(strided_scalar(shape), affine)
        grid = ps.Grid.from_image(img)
        grid_coords = grid.get_coords()

        assert grid.n_coords == prod(shape[:3])
        assert grid.dim == 3
        assert np.allclose(grid.affine, affine)

        assert np.allclose(grid_coords[0], [0, 0, 0])
        # Final index is [4, 4, 4], scaled by affine
        assert np.allclose(grid_coords[-1], [8, 12, 16])

    def test_from_mask(self):
        affine = np.diag([2, 3, 4, 1])
        mask = np.zeros((3, 3, 3))
        mask[1, 1, 1] = 1
        img = SpatialImage(mask, affine)

        grid = ps.Grid.from_mask(img)
        grid_coords = grid.get_coords()

        assert grid.n_coords == 1
        assert grid.dim == 3
        assert np.array_equal(grid_coords, [[2, 3, 4]])

    def test_to_mask(self):
        coords = np.array([[1, 1, 1]])

        grid = ps.Grid(coords)

        mask_img = grid.to_mask()
        assert mask_img.shape == (2, 2, 2)
        assert np.array_equal(mask_img.get_fdata(), [[[0, 0], [0, 0]], [[0, 0], [0, 1]]])
        assert np.array_equal(mask_img.affine, np.eye(4))

        mask_img = grid.to_mask(shape=(3, 3, 3))
        assert mask_img.shape == (3, 3, 3)
        assert np.array_equal(
            mask_img.get_fdata(),
            [
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            ],
        )
        assert np.array_equal(mask_img.affine, np.eye(4))
