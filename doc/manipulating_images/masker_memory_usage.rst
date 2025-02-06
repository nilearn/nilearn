.. _masker_memory_usage:

============================================================================
Beyond the masker: masking larger images, in parallel but memory-efficiently
============================================================================

This chapter discusses a parallelized workflow which can be pretty
memory-intensive when using the masker objects.

Problem
===========

Particularly, we will consider a case where we have a large fMRI image
and we want to extract the data from several regions of interest (ROIs) defined
by a number of binary masks, all in parallel.

The issue when trying to process a large fMRI image in parallel like this is
that each parallel process will load the entire fMRI image into memory. This
can lead to a significant increase in memory usage and can slow down the
processing.


Experiment
==========

So we will try to do exactly what we just described: extract data from several
regions of interest (ROIs) defined by a number of binary masks,
all in parallel. But we will compare three different ways of doing this:

1. Using the :class:`~nilearn.maskers.NiftiMasker`
2. Using numpy indexing
3. Using numpy indexing with shared memory

For first two of these methods, we can provide fMRI image via a file path or
an in-memory image. So we will also compare the memory usage of these two
variations for those methods.


Method
======

We will use `memory_profiler
<https://github.com/pythonprofilers/memory_profiler>`_'s command-line tool
``mprof`` to measure the memory of the whole script.

.. code-block:: bash

        mprof run --include-children --multiprocess script.py

After it has finished running, you can generate a plot showing the memory
usage of the script over time.

.. code-block:: bash

        mprof plot

To resolve the memory usage peaks for each method, we will add 30 second sleep
between each method. This will allow us to see the memory usage of each method
separately in the plot.

Here's the script we will use:

.. code-block:: python

        # script.py
        import time
        from multiprocessing.shared_memory import SharedMemory
        from pathlib import Path

        import numpy as np
        from joblib import Parallel, delayed
        import nibabel as nib

        from nilearn.datasets import fetch_atlas_difumo, fetch_development_fmri
        from nilearn.image import (
            concat_imgs,
            index_img,
            iter_img,
            load_img,
            resample_to_img,
        )
        from nilearn.maskers import NiftiMasker
        from memory_profiler import memory_usage


        def get_fmri_path(n_subjects=1):
            fmri_data = fetch_development_fmri(n_subjects=n_subjects)
            concat = concat_imgs(fmri_data.func)
            Path("temp").mkdir(parents=True, exist_ok=True)
            fmri_path = Path("temp", "fmri.nii.gz")
            concat.to_filename(fmri_path)
            return concat, fmri_path


        def get_atlas_path():
            atlas = fetch_atlas_difumo(dimension=64)
            atlas_path = atlas.maps
            return atlas_path


        def atlas_to_masks(atlas_path, fmri_path, n_regions=6):
            masks = load_img(atlas_path)
            # only keep the first 6 regions
            masks = index_img(masks, slice(0, n_regions))
            mask_paths = []
            resampled_masks = []
            Path("temp").mkdir(parents=True, exist_ok=True)
            for i, mask in enumerate(iter_img(masks)):
                resampled_mask = resample_to_img(
                    mask,
                    fmri_path,
                    interpolation="nearest",
                    copy_header=True,
                    force_resample=True,
                )
                path = Path("temp", f"mask_{i}.nii.gz")
                data = resampled_mask.get_fdata()
                data[data != 0] = 1
                resampled_mask = resampled_mask.__class__(
                    data, resampled_mask.affine, resampled_mask.header
                )
                resampled_mask.to_filename(path)
                mask_paths.append(path)
                resampled_masks.append(resampled_mask)
            return resampled_masks, mask_paths


        def nifti_masker_single(fmri_path, mask_path):
            return NiftiMasker(mask_img=mask_path).fit_transform(fmri_path)


        def numpy_masker_single_path(fmri_path, mask_path):
            return np.asarray(nib.load(fmri_path).dataobj)[
                np.asarray(nib.load(mask_path).dataobj).astype(bool)
            ]


        def numpy_masker_single_inmemory(fmri_img, mask_img):
            return np.asarray(fmri_img.dataobj)[
                np.asarray(mask_img.dataobj).astype(bool)
            ]


        def numpy_masker_shared_single(img, mask):
            return img[np.asarray(mask.dataobj).astype(bool)]


        def nifti_masker_parallel(fmri_path, mask_paths, n_regions=6):
            return Parallel(n_jobs=n_regions)(
                delayed(nifti_masker_single)(fmri_path, mask) for mask in mask_paths
            )


        def numpy_masker_parallel_path(fmri_path, mask_paths, n_regions=6):
            return Parallel(n_jobs=n_regions)(
                delayed(numpy_masker_single_path)(fmri_path, mask)
                for mask in mask_paths
            )


        def numpy_masker_parallel_inmemory(fmri_img, mask_imgs, n_regions=6):
            return Parallel(n_jobs=n_regions)(
                delayed(numpy_masker_single_inmemory)(fmri_img, mask)
                for mask in mask_imgs
            )


        def numpy_masker_shared_parallel(img, masks, n_regions=6):
            return Parallel(n_jobs=n_regions)(
                delayed(numpy_masker_shared_single)(img, mask) for mask in masks
            )


        def main(n_images=1, n_regions=6):
            """
            Compare the performance of NiftiMasker vs. numpy masking vs.
            numpy masking + shared memory both with single and
            `n_regions` parallel processes.

            The first two methods can be used with either file paths
            or in-memory images. So we also compare their memory usage.

            We add 30 second sleep between each method to see the memory usage
            of each method separately in the plot.

            Steps:

            1. fetch `n_images` subjects from development fMRI dataset and
            `n_regions` regions from the Difumo atlas.
            2. convert these regions to binary masks and resample them to the
            fMRI data.
            3. run the following methods in sequence:
                - NiftiMasker with single nifti file path
                - NiftiMasker with single in-memory nifti image
                - NiftiMasker with parallel nifti file paths
                - NiftiMasker with parallel in-memory nifti images
                - numpy masking with single nifti file path
                - numpy masking with single in-memory nifti image
                - numpy masking with parallel nifti file paths
                - numpy masking with parallel in-memory nifti images
                - numpy masking with nifti image in-memory shared by parallel
                processes


            Parameters
            ----------
            n_images : int, default=1
                Number of subjects to fetch from the development fMRI dataset. These
                subject images would be concatenated to form a single nifti file.
                Can be increased to simulate larger data.

            n_regions : int, default=6
                Number of regions to fetch from the Difumo atlas. These regions would
                be converted to binary masks and used to mask the fMRI data. This is
                also the number of jobs to run in parallel.
            """
            fmri_img, fmri_path = get_fmri_path(n_subjects=n_images)
            atlas_path = get_atlas_path()
            mask_imgs, mask_paths = atlas_to_masks(
                atlas_path, fmri_path, n_regions=n_regions
            )

            print("waiting")
            time.sleep(30)
            print("start single nifti masker with path")

            nifti_masker_single(fmri_path, mask_paths[0])

            print("waiting")
            time.sleep(30)
            print("start single nifti masker with in memory images")

            nifti_masker_single(fmri_img, mask_imgs[0])

            print("waiting")
            time.sleep(30)
            print("start parallel nifti masker with paths")

            nifti_masker_parallel(fmri_path, mask_paths, n_regions=n_regions)

            print("waiting")
            time.sleep(30)
            print("start parallel nifti masker with in memory images")

            nifti_masker_parallel(fmri_img, mask_imgs, n_regions=n_regions)

            print("waiting")
            time.sleep(30)
            print("start single numpy masker with path")

            numpy_masker_single_path(fmri_path, mask_paths[0])

            print("waiting")
            time.sleep(30)
            print("start single numpy masker with in memory image")

            numpy_masker_single_inmemory(fmri_img, mask_imgs[0])

            print("waiting")
            time.sleep(30)
            print("start parallel numpy masker with paths")

            numpy_masker_parallel_path(fmri_path, mask_paths, n_regions=n_regions)

            print("waiting")
            time.sleep(30)
            print("start parallel numpy masker with memory image")

            numpy_masker_parallel_inmemory(fmri_img, mask_imgs, n_regions=n_regions)

            print("waiting")
            time.sleep(30)
            print("load image in shared memory")

            fmri_data = np.asarray(fmri_img.dataobj)
            shm = SharedMemory(create=True, size=fmri_data.nbytes)
            shared_data = np.ndarray(
                fmri_data.shape, dtype=fmri_data.dtype, buffer=shm.buf
            )
            np.copyto(shared_data, fmri_data)
            del fmri_data

            print("waiting")
            time.sleep(30)
            print("start parallel numpy masker with shared memory")
            numpy_masker_shared_parallel(shared_data, mask_imgs, n_regions=n_regions)

            shm.close()
            shm.unlink()


        if __name__ == "__main__":
            main(n_images=10, n_regions=20)


Result
======

.. image:: ../images/mprofile_n10_j20_edited.png
    :align: center
    :width: 200%
