function to_canonical(imgs, vox_sizes, prefix, hold)
% Resample images to canonical (transverse) orientation with given voxel sizes
%
% Inspired by ``reorient.m`` by John Ashburner:
% http://blogs.warwick.ac.uk/files/nichols/reorient.m
%
% Parameters
% ----------
% imgs : char or cell array or struct array
%     Images to resample to canonical orientation.
% vox_sizes : vector (3, 1), optional
%     Voxel sizes for output image.
% prefix : char, optional
%     Prefix for output resampled images, default = 'r'
% hold : float, optional
%     Hold (resampling method) value, default = 3.

if ~isstruct(imgs)
    imgs = spm_vol(imgs);
end
if nargin < 2
    vox_sizes = [1 1 1];
elseif numel(vox_sizes) == 1
    vox_sizes = [vox_sizes vox_sizes vox_sizes];
end
vox_sizes = vox_sizes(:);
if nargin < 3
    prefix = 'r';
end
if nargin < 4
    hold = 3;
end

for vol_no = 1:numel(imgs)
    vol = imgs{vol_no}(1);
    % From:
    % http://stackoverflow.com/questions/4165859/generate-all-possible-combinations-of-the-elements-of-some-vectors-cartesian-pr
    sets = {[1, vol.dim(1)], [1, vol.dim(2)], [1, vol.dim(3)]};
    [x y z] = ndgrid(sets{:});
    corners = [x(:) y(:) z(:)];
    corner_coords = [corners ones(length(corners), 1)]';
    corner_mm = vol.mat * corner_coords;
    min_xyz = min(corner_mm(1:3, :), [], 2);
    max_xyz = max(corner_mm(1:3, :), [], 2);
    % Make output volume
    out_vol = vol;
    out_vol.private = [];
    out_vol.mat = diag([vox_sizes' 1]);
    out_vol.mat(1:3, 4) = min_xyz - vox_sizes;
    out_vol.dim(1:3) = ceil((max_xyz - min_xyz) ./ vox_sizes) + 1;
    [dpath, froot, ext] = fileparts(vol.fname);
    out_vol.fname = fullfile(dpath, [prefix froot ext]);
    out_vol = spm_create_vol(out_vol);
    % Resample original volume at output volume grid
    plane_size = out_vol.dim(1:2);
    for slice_no = 1:out_vol.dim(3)
        resamp_affine = inv(spm_matrix([0 0 -slice_no]) * inv(out_vol.mat) * vol.mat);
        slice_vals = spm_slice_vol(vol, resamp_affine, plane_size, hold);
        out_vol = spm_write_plane(out_vol, slice_vals, slice_no);
    end
end
