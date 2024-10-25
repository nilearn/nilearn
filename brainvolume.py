import numpy as np
import nibabel as nib

def compute_brain_region_volume(mask_file, voxel_dimensions):
    """
    Compute the volume of a brain region based on a binary mask.

    Parameters:
    - mask_file (str): Path to the NIfTI file containing the binary brain mask.
    - voxel_dimensions (tuple): Size of the voxels in millimeters (x, y, z).

    Returns:
    - float: Volume of the brain region in cubic millimeters.
    """
    # Load the binary mask using nibabel
    mask_image = nib.load(mask_file)
    mask_data = mask_image.get_fdata()

    # Count the number of non-zero voxels in the mask
    non_zero_voxel_count = np.sum(mask_data > 0)

    # Calculate the volume by multiplying the count of non-zero voxels by the volume of each voxel
    total_volume = non_zero_voxel_count * np.prod(voxel_dimensions)  # in cubic mm

    return total_volume

# Example of how to use the function
if __name__ == "__main__":
    # Specify the path to a binary mask and the voxel dimensions (e.g., [2, 2, 2] mm)
    mask_file_path = "path/to/your/mask.nii"
    voxel_dimensions = (2.0, 2.0, 2.0)  # Example voxel size in mm

    # Calculate the volume of the brain region
    brain_region_volume = compute_brain_region_volume(mask_file_path, voxel_dimensions)
    print(f"Computed brain region volume: {brain_region_volume} cubic mm")
