"""Drafting."""

from nilearn.experimental import surface

# from nilearn.maskers import MultiNiftiMasker, NiftiMasker

# for i, masker_func in  enumerate([NiftiMasker, MultiNiftiMasker]):
#     masker = masker_func()
#     report = masker.generate_report()
#     report.save_as_html(f"masker_empty_{i}.html")


# labels_img, label_names = surface.fetch_destrieux()
# label = 62
# for part in labels_img.data.parts:
#     tmp = labels_img.data.parts[part] == label
#     labels_img.data.parts[part] = tmp.astype(int)

# masker = surface.SurfaceMasker()
# img = surface.fetch_nki(mesh_type="inflated", n_subjects=1)[0]
# masker.fit_transform(img)
# report = masker.generate_report()
# report.save_as_html("surface_masker_nki.html")


labels_img, label_names = surface.fetch_destrieux(mesh_type="inflated")

labels_masker = surface.SurfaceLabelsMasker(labels_img, label_names).fit()
report = labels_masker.generate_report()
report.save_as_html("surface_label_masker_unfitted.html")

# masked_data = labels_masker.transform(img)
# report = labels_masker.generate_report()
# report.save_as_html("surface_label_masker.html")
