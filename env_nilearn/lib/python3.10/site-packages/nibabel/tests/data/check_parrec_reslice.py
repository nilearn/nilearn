"""Test differences in affines by reslicing

Should be run from directory containing .PAR _and_ matching .REC files from
Michael's PAR / REC dataset at:

    http://psydata.ovgu.de/philips_achieva_testfiles/conversion2

Gives output something like:

RMS of standard image Phantom_EPI_3mm_tra_SENSE_6_1.PAR           : 148.619965177
RMS resliced Phantom_EPI_3mm_cor_20APtrans_15RLrot_SENSE_15_1.PAR : 22.0706570007
RMS resliced Phantom_EPI_3mm_cor_SENSE_8_1.PAR                    : 47.5762615987
RMS resliced Phantom_EPI_3mm_sag_15AP_SENSE_13_1.PAR              : 25.0972161667
RMS resliced Phantom_EPI_3mm_sag_15FH_SENSE_12_1.PAR              : 28.7508166372
RMS resliced Phantom_EPI_3mm_sag_15RL_SENSE_11_1.PAR              : 29.0544513507
RMS resliced Phantom_EPI_3mm_sag_SENSE_7_1.PAR                    : 25.7621452929
RMS resliced Phantom_EPI_3mm_tra_-30AP_10RL_20FH_SENSE_14_1.PAR   : 32.0602533689
RMS resliced Phantom_EPI_3mm_tra_15FH_SENSE_9_1.PAR               : 28.8953071672
RMS resliced Phantom_EPI_3mm_tra_15RL_SENSE_10_1.PAR              : 29.0793602478

The *_cor_SENSE* image has a higher RMS because the back of the phantom is out
of the field of view.
"""

import glob

import numpy as np
import numpy.linalg as npl

import nibabel as nib
from nibabel import parrec
from nibabel.affines import to_matvec
from nibabel.optpkg import optional_package

_, have_scipy, _ = optional_package('scipy')


def resample_img2img(img_to, img_from, order=1, out_class=nib.Nifti1Image):
    if not have_scipy:
        raise Exception('Scipy must be installed to run resample_img2img.')

    from scipy import ndimage as spnd

    vox2vox = npl.inv(img_from.affine).dot(img_to.affine)
    rzs, trans = to_matvec(vox2vox)
    data = spnd.affine_transform(img_from.get_fdata(), rzs, trans, img_to.shape, order=order)
    return out_class(data, img_to.affine)


def gmean_norm(data):
    in_data = data > np.mean(data) * 0.8
    gmean = np.mean(data[in_data])
    return data / gmean


if __name__ == '__main__':
    np.set_printoptions(suppress=True, precision=4)
    normal_fname = 'Phantom_EPI_3mm_tra_SENSE_6_1.PAR'
    normal_img = parrec.load(normal_fname)
    normal_data = normal_img.get_fdata()
    normal_normed = gmean_norm(normal_data)

    print(f'RMS of standard image {normal_fname:<44}: {np.sqrt(np.sum(normal_normed ** 2))}')

    for parfile in glob.glob('*.PAR'):
        if parfile == normal_fname:
            continue
        funny_img = parrec.load(parfile)
        fixed_img = resample_img2img(normal_img, funny_img)
        fixed_data = fixed_img.get_fdata()
        difference_data = normal_normed - gmean_norm(fixed_data)
        print(f'RMS resliced {parfile:<52} : {np.sqrt(np.sum(difference_data ** 2))}')
