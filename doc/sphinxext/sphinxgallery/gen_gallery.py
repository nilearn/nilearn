# -*- coding: utf-8 -*-
"""
Created on Wed Jan 28 10:38:13 2015

@author: oscar
"""
import os
from sphinxgallery.gen_rst import generate_dir_rst
from sphinxgallery.docs_resolv import embed_code_links


def generate_gallery_rst(app):
    """ Generate the list of examples, as well as the contents of
        examples.
    """
    try:
        plot_gallery = eval(app.builder.config.plot_gallery)
    except TypeError:
        plot_gallery = bool(app.builder.config.plot_gallery)

    if not plot_gallery:
        return

    input_conf = app.config.sphinxgallery_conf
    gallery_conf = DEFAULT_CONF
    for conf, value in input_conf.items():
        gallery_conf[conf] = value

    app.config.sphinxgallery_conf = gallery_conf # this assures I can call the config in other places

    root_dir = os.path.join(app.builder.srcdir, gallery_conf['root_dir'])
    gallery_dir = os.path.join(app.builder.srcdir, gallery_conf['examples_gallery'])
    generated_dir = os.path.join(app.builder.srcdir, gallery_conf['mod_generated'])

    for workdir in [root_dir, gallery_dir, generated_dir]:
        if not os.path.exists(workdir):
            os.makedirs(workdir)

    # we create an index.rst with all examples
    fhindex = open(os.path.join(gallery_dir, 'index.rst'), 'w')
    fhindex.write("""

.. _examples-index:

Examples
========

""")
    # Here we don't use an os.walk, but we recurse only twice: flat is
    # better than nested.
    seen_backrefs = set()
    generate_dir_rst('.', fhindex, root_dir, gallery_dir, gallery_conf, plot_gallery, seen_backrefs)
    for directory in sorted(os.listdir(root_dir)):
        if os.path.isdir(os.path.join(root_dir, directory)):
            generate_dir_rst(directory, fhindex, root_dir, gallery_dir, gallery_conf, plot_gallery, seen_backrefs)
    fhindex.flush()


DEFAULT_CONF = {
    'root_dir'          : '../examples',
    'examples_gallery'  : 'auto_examples',
    'mod_generated'     : 'modules/generated',
    'doc_module'        : 'sphinxgallery',
    'resolver_urls'     : {
        'matplotlib': 'http://matplotlib.org',
        'numpy': 'http://docs.scipy.org/doc/numpy-1.9.1',
        'scipy': 'http://docs.scipy.org/doc/scipy-0.15.1/reference'}
    }

def setup(app):
    app.add_config_value('plot_gallery', True, 'html')
    app.add_config_value('sphinxgallery_conf', DEFAULT_CONF, 'html')
    app.add_stylesheet('gallery.css')

    app.connect('builder-inited', generate_gallery_rst)

    app.connect('build-finished', embed_code_links)

    # Sphinx hack: sphinx copies generated images to the build directory
    #  each time the docs are made.  If the desired image name already
    #  exists, it appends a digit to prevent overwrites.  The problem is,
    #  the directory is never cleared.  This means that each time you build
    #  the docs, the number of images in the directory grows.
    #
    # This question has been asked on the sphinx development list, but there
    #  was no response: http://osdir.com/ml/sphinx-dev/2011-02/msg00123.html
    #
    # The following is a hack that prevents this behavior by clearing the
    #  image build directory each time the docs are built.  If sphinx
    #  changes their layout between versions, this will not work (though
    #  it should probably not cause a crash).  Tested successfully
    #  on Sphinx 1.0.7
    build_image_dir = '_build/html/_images'
    if os.path.exists(build_image_dir):
        filelist = os.listdir(build_image_dir)
        for filename in filelist:
            if filename.endswith('png'):
                os.remove(os.path.join(build_image_dir, filename))

def setup_module():
    # HACK: Stop nosetests running setup() above
    pass