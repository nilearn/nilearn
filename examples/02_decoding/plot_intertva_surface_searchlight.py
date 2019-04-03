"""
Cortical searchlight analysis of frequencies in grabbr experiment
=================================================================

Searchlight analysis requires fitting a classifier a large amount of
times. As a result, it is an intrinsically slow method. In order to speed
up computing, in this example, Searchlight is run only on one slice on
the fMRI (see the generated figures).

"""

### Load grabbr dataset ########################################################
import numpy as np
import nibabel.gifti as ng
import nibabel as nb
import os.path as op
import os
import sys
import pandas

from sklearn.linear_model import LogisticRegression

# to use dev version of nilearn...
sys.path.insert(1,"/envau/userspace/takerkart/python/tmp_nilearn_surfsearchlight_test/nilearn")


root_dir = '/hpc/banco/cagna.b/my_intertva/surf/data'

#sub-30/glm/surf/usub-30_task-localizer_model-singletrial_denoised/

#root_dir = '/riou/work/crise/takerkart/grabbr'
#root_dir = '/hpc/crise/takerkart/oldgrabbr'
#fs_db_dir = op.join(root_dir,'sanlm_fs_db')
fs_db_dir = '/hpc/banco/cagna.b/my_intertva/surf/results/searchlight/surf'
#surfsmooth = 3
#volsmooth = 0

verbose = 0

preprocessing_prefix = 'u'
glm_type = 'glm'
modelname = 'model-singletrial_denoised'
task = 'task-localizer'


#n_sess = 5
n_trials = 144


#def run_searchlight(subject,radius,hemroi):

if True:

    subject = 'sub-41'
    radius = 12
    hemroi = {'hem':'lh', 'roi':'fullbrain'}

    #surfglm_dir = op.join(root_dir,subject,glm_type,modelname,'fsavg5.ss%02d_vs%02d' % (surfsmooth,volsmooth),'betas')
    surfglm_dir = op.join(root_dir,subject,glm_type,'surf','{}{}_{}_{}'.format(preprocessing_prefix, subject, task, modelname))
    print(surfglm_dir)
    #sub-30/glm/surf/usub-30_task-localizer_model-singletrial_denoised/

    labels_df = pandas.read_csv("/envau/work/banco/InterTVA/zenodo/labels_voicelocalizer_voice_vs_nonvoice.tsv", sep='\t')
    y = np.array(labels_df['label'])

    # load spherical mesh
    #mesh_path = op.join(fs_db_dir,'fsaverage_gii','%s.sphere.gii' % hemroi['hem'])
    mesh_path = op.join('/hpc/soft/freesurfer/gifti_fsaverage_meshes/fsaverage5.%s.sphere.surf.gii' % hemroi['hem'])
    orig_mesh = nb.load(mesh_path)
    
    print(hemroi['roi'])
    
    
    # if hemroi['roi'] != 'fullbrain':
    #     # load cortical roi
    #     #hemroi_path = op.join(fs_db_dir,'fsaverage5.%s.%s.gii' % (hemroi['hem'], hemroi['roi']))
    #     hemroi_path = '/hpc/banco/cagna.b/my_intertva/surf/results/searchlight/surf/fsaverage5.{}.u_task-localizer_model-singletrial_denoised_class-vnv_r16/stats/spmT_0001_unc05_e010_bin.gii'.format(hemroi['hem'])
    #     print('bouhhhhhhhhhhhhh')
    #     roi_surfmask_tex = nb.load(hemroi_path)
    # else:
    #     # really bad trick to be fixed later!!!!!!!!!!
    #     #hemroi_path = op.join(fs_db_dir,'fsaverage_gii','fsaverage5.%s.%s.gii' % (hemroi['hem'], 'auditory_cortex'))
    #     hemroi_path = '/hpc/banco/cagna.b/my_intertva/surf/results/searchlight/surf/fsaverage5.{}.u_task-localizer_model-singletrial_denoised_class-vnv_r16/stats/spmT_0001_unc05_e010_bin.gii'.format(hemroi['hem'])
    #     roi_surfmask_tex = nb.load(hemroi_path)
    
    '''
    hemroi_path = op.join(fs_db_dir,'fsaverage_gii','fsaverage5.%s.%s.gii' % (hemroi['hem'], hemroi['roi']))
    print hemroi_path
    roi_surfmask_tex = nb.load(hemroi_path)
    '''
    
    alldata_singletrials = []
    sessions_singletrials = []
    tex_path_list = []
    single_counter = 0
    for current_trial in range(n_trials):
        tex_path = op.join(surfglm_dir, 'fsaverage5.{}.beta_{:04d}.gii'.format(hemroi['hem'],current_trial+1))
        #print tex_path
        tex = nb.load(tex_path)
        alldata_singletrials.append(tex.darrays[0].data)
        #y_singletrials.append(f+1)
        #sessions_singletrials.append(sess+1)
        #single_counter = single_counter + 1
    
    data = np.array(alldata_singletrials).T
    print(data.shape)
    print('Number of Nans: {:d}'.format(np.count_nonzero(np.isnan(data))))
    data[np.isnan(data)] = 0
    # compute the full brain mask
    surfmask = data.sum(1).astype(np.float32)
    surfmask[surfmask!=0] = 1.
    # trick it: re-read the roi mask and copy the data into the new gii object
    #surfmask_tex = nb.load(hemroi_path)
    surfmask_tex = nb.load(tex_path)
    surfmask_tex.darrays[0].data = surfmask
    print(np.flatnonzero(surfmask).size)
    surfmask_path = op.join(surfglm_dir, 'fsaverage5.%s.brainmask.gii' % hemroi['hem'])
    ng.write(surfmask_tex,surfmask_path)
    
    ### Searchlight computation ###################################################
    
    # Make processing parallel
    n_jobs = 12
    
    ### Define the cross-validation scheme used for validation.
    # Here we use a LeaveOneSessionOut cross validation
    #from sklearn.cross_validation import LeaveOneLabelOut
    #lolo = LeaveOneLabelOut(sessions_singletrials)
    
    from sklearn.model_selection import StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=100, test_size=0.15)
    
    logreg = LogisticRegression()

    import nilearn.decoding
    # The radius is the one of the Searchlight sphere that will scan the volume
    
    if hemroi['roi'] == 'fullbrain':
        searchlight = nilearn.decoding.SurfSearchLight(orig_mesh,
                                                       surfmask_tex,
                                                       estimator=logreg,
                                                       process_surfmask_tex=surfmask_tex,
                                                       radius = radius,
                                                       verbose = 1,
                                                       n_jobs=n_jobs,
                                                       cv=sss)
        #,
        #                                           cv = lolo)
        #n_jobs = n_jobs,
    else:
        searchlight = nilearn.decoding.SurfSearchLight(orig_mesh,
                                                       surfmask_tex,
                                                       process_surfmask_tex=roi_surfmask_tex,
                                                       radius = radius,
                                                       n_jobs = n_jobs,
                                                       verbose = 1)
        # ,
        #                                                cv = lolo)
        
    searchlight.fit(data, y)
    
    # trick it: re-read the roi mask and copy the searchlight scores to save the gii object
    slscores_tex = nb.load(tex_path)
    slscores_tex.darrays[0].data = searchlight.scores_.astype(np.float32)
    slscores_path = op.join(surfglm_dir, 'fsaverage5.%s.res.gii' % hemroi['hem'])
    ng.write(slscores_tex,slscores_path)




    # slres_dir = op.join(root_dir,subject,glm_type,modelname,'fsavg5.ss%02d_vs%02d' % (surfsmooth,volsmooth),'tmp_searchlight_test')
    # if not(op.exists(slres_dir)):
    #     os.makedirs(slres_dir)
    #     print('Creating new directory: %s' % slres_dir)
    # else:
    #     print('Warning: overwriting content in existing directory: %s' % slres_dir)
    # #slscores_path = op.join(slres_dir,'%s.%s.sl_cl%s_rad%1.1f.gii' % (hemroi['hem'],hemroi['roi'],freqs_string,radius))
    # slscores_path = op.join(slres_dir,'%s.%s.sl_cl%s_rad%1.1f.gii' % (hemroi['hem'],'fullbrain',freqs_string,radius))
    # ng.write(slscores_tex,slscores_path)
    

"""
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

"""
