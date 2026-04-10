# Nilearn assets

This directory contains HTML jinja templates, CSS and javascript files
required for building estimator reports
as well as several interactive visualizations (see `view_img`, `view_connectome`, `view_surf`).

``` tree
├── css
│   ├── partials
│   │   └── navbar.css
│   ├── head.css
│   └── report.css
├── html
│   ├── glm
│   │   ├── partials
│   │   │   ├── method_section.jinja
│   │   │   └── navbar.jinja
│   │   └── body_glm.jinja
│   ├── masker
│   │   ├── partials
│   │   │   ├── brainsprite.jinja
│   │   │   ├── brainsprite_opacity.jinja
│   │   │   ├── carousel.jinja
│   │   │   ├── figure.jinja
│   │   │   ├── info_transfom.jinja
│   │   │   └── parameters.jinja
│   │   ├── body_masker.jinja
│   │   ├── body_nifti_labels_masker.jinja
│   │   ├── body_nifti_maps_masker.jinja
│   │   ├── body_nifti_spheres_masker.jinja
│   │   ├── body_surface_maps_masker.jinja
│   │   └── body_surface_masker.jinja
│   ├── partials
│   │   ├── missing_plotting_engine.jinja
│   │   └── warnings.jinja
│   ├── plotting
│   │   ├── connectome_plot.jinja
│   │   ├── surface_plot.jinja
│   │   └── view_img.jinja
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
