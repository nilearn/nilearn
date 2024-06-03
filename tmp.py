"""Drafting."""

from nilearn.experimental import surface

img = surface.fetch_nki()[0]
print(f"NKI image: {img}")

masker = surface.SurfaceMasker()
masked_data = masker.fit_transform(img)
report = masker.generate_report()
report.save_as_html("surface_masker_nki.html")
