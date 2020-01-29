# -*- coding: utf-8 -*-
"""
Adapted from tests authored by Elizabeth Dupre (@emdupre)
 for the Nilearn project in https://github.com/nilearn/nilearn/pull/2019
Inspired from https://github.com/mne-tools/mne-python/
"""
import os.path as op

import numpy as np
from nibabel.tmpdirs import TemporaryDirectory
from sklearn.datasets.base import Bunch

from nistats._utils.testing import _write_fake_fmri_data
from nistats.first_level_model import FirstLevelModel
from nistats.reporting import _ReportScraper, make_glm_report


def _gen_report():
    """ Generate an empty HTMLReport for testing """

    shapes, rk = ((7, 8, 7, 15), (7, 8, 7, 16)), 3
    mask, fmri_data, design_matrices = _write_fake_fmri_data(shapes, rk)
    flm = FirstLevelModel(mask_img=mask).fit(
            fmri_data, design_matrices=design_matrices)
    contrast = np.eye(3)[1]
    report = make_glm_report(flm, contrast, plot_type='glass',
                             height_control=None, min_distance=15,
                             alpha=0.001, threshold=2.78,
                             )
    return report


def test_scraper():
    """Test report scraping."""
    # Mock a Sphinx + sphinx_gallery config
    with TemporaryDirectory() as tmpdir:
        app = Bunch(builder=Bunch(srcdir=str(tmpdir),
                                  outdir=op.join(str(tmpdir),
                                                 '_build', 'html')
                                  ))
        scraper = _ReportScraper()
        scraper.app = app
        gallery_conf = dict(src_dir=app.builder.srcdir, builder_name='html')
        img_fname = op.join(app.builder.srcdir, 'auto_examples', 'images',
                            'sg_img.png')
        target_file = op.join(app.builder.srcdir, 'auto_examples', 'sg.py')
        block_vars = dict(image_path_iterator=(img for img in [img_fname]),
                          example_globals=dict(a=1), target_file=target_file)
        # Confirm that HTML isn't accidentally inserted
        block = None
        rst = scraper(block, block_vars, gallery_conf)
        assert rst == ''

        # Confirm that HTML is correctly inserted for HTMLReport
        report = _gen_report()
        block_vars['example_globals']['report'] = report
        rst = scraper(block, block_vars, gallery_conf)
        # assert "<detail" in rst
        assert 'Model details:' in rst
        assert 'Design Matrix:' in rst
        assert 'Contrasts' in rst
        assert 'Mask' in rst
        assert 'Stat Maps with Cluster Tables' in rst
        assert 'Contrast Plot' in rst
        assert 'cluster-details-table' in rst
        assert 'Cluster Table' in rst
