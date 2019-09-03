import tempfile
import numpy as np
import os.path as op
from nibabel import Nifti1Image
from sklearn.utils import Bunch
from nilearn.input_data import NiftiMasker
from nilearn.reporting import _ReportScraper

def _gen_report():
    """ Generate an empty HTMLReport for testing """

    data = np.zeros((9, 9, 9))
    data[3:-3, 3:-3, 3:-3] = 10
    data_img_3d = Nifti1Image(data, np.eye(4))

    # turn off reporting
    mask = NiftiMasker()
    mask.fit(data_img_3d)
    report = mask.generate_report()
    return report


def test_scraper():
    """Test report scraping."""
    tmpdir = tempfile.mkdtemp()
    # Mock a Sphinx + sphinx_gallery config
    app = Bunch(builder=Bunch(srcdir=str(tmpdir),
                              outdir=op.join(str(tmpdir), '_build', 'html')))
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
    assert "<detail" in rst
