% Script uses SPM to resample moved anatomical image.
%
% Run `python make_moved_anat.py` to generate file to work on.
%
% Run from the directory containing this file.
% Works with Octave or MATLAB.
% Needs SPM (5, 8 or 12) on the MATLAB path.
P = {'functional.nii', 'anat_moved.nii'};
% Resample without masking
flags = struct('mask', false, 'mean', false, ...
               'interp', 1, 'which', 1, ...
               'prefix', 'resampled_');
spm_reslice(P, flags);
% Reorient to canonical orientation at 4mm resolution, polynomial interpolation
to_canonical({'anat_moved.nii'}, 4, 'reoriented_', 1);
