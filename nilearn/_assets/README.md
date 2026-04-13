# Nilearn assets

This directory contains HTML jinja templates, CSS and javascript files
required for building estimator reports
as well as several interactive visualizations (see `view_img`, `view_connectome`, `view_surf`).

``` tree
├── css
│   ├── partials
│   ├── head.css
│   └── report.css
├── html
│   ├── glm
│   │   ├── partials
│   │   └── body_glm.jinja
│   ├── masker
│   │   ├── partials
│   │   ├── body_masker.jinja
│   │   ├── body_nifti_labels_masker.jinja
│   │   ├── body_nifti_maps_masker.jinja
│   │   ├── body_nifti_spheres_masker.jinja
│   │   ├── body_surface_maps_masker.jinja
│   │   └── body_surface_masker.jinja
│   ├── partials
│   ├── plotting
│   ├── body_base.jinja              # base template for body of estimator reports
│   └── head.jinja                   # common base template for all HTML
├── js
│   ├── brainsprite.min.js           # brainsprite library (https://simexp.github.io/brainsprite.js)
│   ├── carousel.js
│   ├── common-surface-plot-utils.js # common code for `view_connectome`, `view_surf`, 'view_markers`
│   ├── connectome-plot-utils.js
│   ├── jquery.min.js                # jquery library (https://jquery.com/)
│   ├── plotly-gl3d-latest.min.js    # plotly library (https://plot.ly/javascript/getting-started/)
│   └── surface-plot-utils.js
├── __init__.py
└── README.md
```
