"""
Cortical searchlight analysis of frequencies in grabbr experiment
=================================================================

Searchlight analysis requires fitting a classifier a large amount of
times. As a result, it is an intrinsically slow method. In order to speed
up computing, in this example, Searchlight is run only on one slice on
the fMRI (see the generated figures).

The current version (2018/06/14) is a copy of 
/envau/userspace/takerkart/python/tmp_nilearn_surfsearchlight_test/02_cortical_searchlight_python3.py
"""





### Load grabbr dataset ########################################################
import numpy as np
import nibabel.gifti as ng
import nibabel as nb
import os.path as op
import os
import sys

# to use dev version of nilearn...
sys.path.insert(1,"/envau/userspace/takerkart/python/tmp_nilearn_surfsearchlight_test/nilearn")


#root_dir = '/riou/work/crise/takerkart/grabbr'
root_dir = '/hpc/crise/takerkart/oldgrabbr'
fs_db_dir = op.join(root_dir,'sanlm_fs_db')

surfsmooth = 3
volsmooth = 0

verbose = 0

glm_type = 'spm_glms'
modelname = 'freqs_inst_mov_singlebloc'


# all existing conditions in the experiment
allfreqs = ['lowest', 'low', 'mid', 'high', 'highest']
# the conditions to be decoded for this analysis
freqs = ['high', 'highest']
#freqs = ['lowest', 'low']
freqs = ['lowest', 'low', 'mid', 'high', 'highest']
# create freqs_string
f_list = []
for freq in freqs:
    f_list.append(allfreqs.index(freq))
f_list.sort()
freqs_string = ''
for f_ind in f_list:
    freqs_string = freqs_string + str(f_ind)


n_sess = 5
n_trials = 6


def run_searchlight(subject,radius,hemroi):

    surfglm_dir = op.join(root_dir,subject,glm_type,modelname,'fsavg5.ss%02d_vs%02d' % (surfsmooth,volsmooth),'betas')

    # load spherical mesh
    #mesh_path = op.join(fs_db_dir,'fsaverage_gii','%s.sphere.gii' % hemroi['hem'])
    mesh_path = op.join('/hpc/soft/freesurfer/gifti_fsaverage_meshes/fsaverage5.%s.sphere.surf.gii' % hemroi['hem'])
    orig_mesh = nb.load(mesh_path)
    
    print(hemroi['roi'])
    
    
    if hemroi['roi'] != 'fullbrain':
        # load cortical roi
        hemroi_path = op.join(fs_db_dir,'fsaverage_gii','fsaverage5.%s.%s.gii' % (hemroi['hem'], hemroi['roi']))
        print('bouhhhhhhhhhhhhh')
        roi_surfmask_tex = nb.load(hemroi_path)
    else:
        # really bad trick to be fixed later!!!!!!!!!!
        hemroi_path = op.join(fs_db_dir,'fsaverage_gii','fsaverage.%s.%s.gii' % (hemroi['hem'], 'auditory_cortex'))
        roi_surfmask_tex = nb.load(hemroi_path)
    
    '''
    hemroi_path = op.join(fs_db_dir,'fsaverage_gii','fsaverage5.%s.%s.gii' % (hemroi['hem'], hemroi['roi']))
    print hemroi_path
    roi_surfmask_tex = nb.load(hemroi_path)
    '''
    
    alldata_singletrials = []
    y_singletrials = []
    sessions_singletrials = []
    tex_path_list = []
    single_counter = 0
    for sess in range(n_sess):
        for f,freq in enumerate(freqs):
            for t in range(n_trials):
                contrast_id = '%s%02d' % (freq,t+1)
                tex_path = op.join(surfglm_dir, 'fsavg5.%s.func%02d_%s.gii' % (hemroi['hem'],sess+1,contrast_id))
                #print tex_path
                tex = nb.load(tex_path)
                alldata_singletrials.append(tex.darrays[0].data)
                y_singletrials.append(f+1)
                sessions_singletrials.append(sess+1)
                single_counter = single_counter + 1
    
    data = np.array(alldata_singletrials).T
    # compute the full brain mask
    surfmask = data.sum(1).astype(np.float32)
    surfmask[surfmask!=0] = 1.
    # trick it: re-read the roi mask and copy the data into the new gii object
    surfmask_tex = nb.load(hemroi_path)
    surfmask_tex.darrays[0].data = surfmask
    print(np.flatnonzero(surfmask).size)
    surfmask_path = op.join(surfglm_dir, 'fsavg5.%s.brainmask.gii' % hemroi['hem'])
    ng.write(surfmask_tex,surfmask_path)
    
    y = np.array(y_singletrials)
    
    ### Searchlight computation ###################################################
    
    # Make processing parallel
    n_jobs = 16
    
    ### Define the cross-validation scheme used for validation.
    # Here we use a LeaveOneSessionOut cross validation
    from sklearn.cross_validation import LeaveOneLabelOut
    lolo = LeaveOneLabelOut(sessions_singletrials)
    
    
    import nilearn.decoding
    # The radius is the one of the Searchlight sphere that will scan the volume
    
    if hemroi['roi'] == 'fullbrain':
        searchlight = nilearn.decoding.SurfSearchLight(orig_mesh,
                                                   surfmask_tex,
                                                   process_surfmask_tex=surfmask_tex,
                                                   radius = radius,
                                                   n_jobs = n_jobs,
                                                   verbose = 1,
                                                   cv = lolo)
    else:
        searchlight = nilearn.decoding.SurfSearchLight(orig_mesh,
                                                       surfmask_tex,
                                                       process_surfmask_tex=roi_surfmask_tex,
                                                       radius = radius,
                                                       n_jobs = n_jobs,
                                                       verbose = 1,
                                                       cv = lolo)
        
    searchlight.fit(data, y)
    
    # trick it: re-read the roi mask and copy the searchlight scores to save the gii object
    slscores_tex = nb.load(tex_path)
    slscores_tex.darrays[0].data = searchlight.scores_.astype(np.float32)
    slres_dir = op.join(root_dir,subject,glm_type,modelname,'fsavg5.ss%02d_vs%02d' % (surfsmooth,volsmooth),'tmp_searchlight_test')
    if not(op.exists(slres_dir)):
        os.makedirs(slres_dir)
        print('Creating new directory: %s' % slres_dir)
    else:
        print('Warning: overwriting content in existing directory: %s' % slres_dir)
    #slscores_path = op.join(slres_dir,'%s.%s.sl_cl%s_rad%1.1f.gii' % (hemroi['hem'],hemroi['roi'],freqs_string,radius))
    slscores_path = op.join(slres_dir,'%s.%s.sl_cl%s_rad%1.1f.gii' % (hemroi['hem'],'fullbrain',freqs_string,radius))
    ng.write(slscores_tex,slscores_path)
    


def main():
    args = sys.argv[1:]
    subject = args[0]
    radius = float(args[1])
    hem = args[2]
    roi = args[3] # use 'fullbrain' if you want to run the searchlight on all the available data
    hemroi = {'hem':hem, 'roi':roi}
    #hemroi = {'hem':'rh', 'roi':'m1s1'}

    # example: python 02_cortical_searchlight.py grabbr_21 4 lh fullbrain

    #run_searchlight(subject,radius,hemroi)
    
    if subject == 'allsubj':
        # run on the full list
        subjects_list = ['grabbr_03','grabbr_04','grabbr_05','grabbr_06','grabbr_07','grabbr_08','grabbr_09','grabbr_10','grabbr_11','grabbr_12','grabbr_13','grabbr_14','grabbr_15','grabbr_16','grabbr_17','grabbr_18','grabbr_19']
        for subject in subjects_list:
            print(subject)
            run_searchlight(subject,radius,hemroi)
    else:
        run_searchlight(subject,radius,hemroi)
    
    '''
    if len(args) == 0:
        # run on the full list
        subjects_list = ['grabbr_03','grabbr_04','grabbr_05','grabbr_06','grabbr_09','grabbr_10','grabbr_11','grabbr_13','grabbr_14','grabbr_15','grabbr_17','grabbr_19']
        for subject in subjects_list:
            project_beta_maps(subject)
            project_beta_maps_single_trial(subject)
    else:
        # take the first argument as a subject, and run only on this subject
        subject = args[0]
        project_beta_maps(subject)
        project_beta_maps_single_trial(subject)
    '''
 
if __name__ == "__main__":
    main()

