# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
import warnings

import numpy as np

from ...core.api import ImageList


class FmriImageList(ImageList):
    """ Class to implement image list interface for FMRI time series

    Allows metadata such as volume and slice times
    """

    def __init__(self, images=None, volume_start_times=None, slice_times=None):
        """
        An implementation of an fMRI image as in ImageList

        Parameters
        ----------
        images : iterable
           an iterable object whose items are meant to be images; this is
           checked by asserting that each has a `coordmap` attribute and a
           ``get_data`` method.  Note that Image objects are not iterable by
           default; use the ``from_image`` classmethod or ``iter_axis`` function
           to convert images to image lists - see examples below for the latter.
        volume_start_times: None or float or (N,) ndarray
            start time of each frame. It can be specified either as an ndarray
            with ``N=len(images)`` elements or as a single float, the TR. None
            results in ``np.arange(len(images)).astype(np.float)``
        slice_times: None or (N,) ndarray
            specifying offset for each slice of each frame, from the frame start
            time.

        See Also
        --------
        nipy.core.image_list.ImageList

        Examples
        --------
        >>> from nipy.testing import funcfile
        >>> from nipy.io.api import load_image
        >>> from nipy.core.api import iter_axis
        >>> funcim = load_image(funcfile)
        >>> iterable_img = iter_axis(funcim, 't')
        >>> fmrilist = FmriImageList(iterable_img)
        >>> print fmrilist.get_list_data(axis=0).shape
        (20, 17, 21, 3)
        >>> print fmrilist[4].shape
        (17, 21, 3)
        """
        ImageList.__init__(self, images=images)
        if volume_start_times is None:
            volume_start_times = 1.
        v = np.asarray(volume_start_times)
        length = len(self.list)
        if v.shape == (length,):
            self.volume_start_times = volume_start_times
        else:
            v = float(volume_start_times)
            self.volume_start_times = np.arange(length) * v
        self.slice_times = slice_times

    def __getitem__(self, index):
        """
        If index is an index, return self.list[index], an Image
        else return an FmriImageList with images=self.list[index].
        """
        if type(index) is type(1):
            return self.list[index]
        return self.__class__(
            images=self.list[index],
            volume_start_times=self.volume_start_times[index],
            slice_times=self.slice_times)

    @classmethod
    def from_image(klass, fourdimage, axis='t',
                   volume_start_times=None, slice_times=None):
        """Create an FmriImageList from a 4D Image

        Get images by extracting 3d images along the 't' axis.

        Parameters
        ----------
        fourdimage : ``Image`` instance
            A 4D Image
        volume_start_times: None or float or (N,) ndarray
            start time of each frame. It can be specified either as an ndarray
            with ``N=len(images)`` elements or as a single float, the TR. None
            results in ``np.arange(len(images)).astype(np.float)``
        slice_times: None or (N,) ndarray
            specifying offset for each slice of each frame, from the frame start
            time.

        Returns
        -------
        filist : ``FmriImageList`` instance
        """
        if fourdimage.ndim != 4:
            raise ValueError('expecting a 4-dimensional Image')
        image_list = ImageList.from_image(fourdimage, axis)
        return klass(images=image_list.list,
                     volume_start_times=volume_start_times,
                     slice_times=slice_times)


def axis0_generator(data, slicers=None):
    """ Takes array-like `data`, returning slices over axes > 0

    This function takes an array-like object `data` and yields tuples of slicing
    thing and slices like::

        [slicer, np.asarray(data)[:,slicer] for slicer in slicer]

    which in the default (`slicers` is None) case, boils down to::

        [i, np.asarray(data)[:,i] for i in range(data.shape[1])]

    This can be used to get arrays of time series out of an array if the time
    axis is axis 0.

    Parameters
    ----------
    data : array-like
       object such that ``arr = np.asarray(data)`` returns an array of
       at least 2 dimensions.
    slicers : None or sequence
       sequence of objects that can be used to slice into array ``arr``
       returned from data.  If None, default is ``range(data.shape[1])``
    """
    arr = np.asarray(data)
    if slicers is None:
        slicers = range(arr.shape[1])
    for slicer in slicers:
        yield slicer, arr[:,slicer]

