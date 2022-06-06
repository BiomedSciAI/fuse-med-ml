import os
import pandas as pd
import numpy as np
import SimpleITK as sitk
from datetime import datetime,timedelta
from typing import Hashable, Optional, Sequence
from tqdm import tqdm
from fuse.utils import NDict
from fuseimg.datasets import duke
from fuse.data.utils.sample import create_initial_sample
from fuseimg.data.ops import ops_mri
from fuseimg.datasets.duke import Duke,get_duke_raw_annotations_df
from fuse.data import PipelineDefault
import pickle as pkl
from radiomics import featureextractor  # This module is used for interaction with pyradiomics

def get_selected_series_index(sample_id, seq_id):
    patient_id = sample_id[0]
    if patient_id in ['Breast_MRI_120', 'Breast_MRI_596']:
        map = {'DCE_mix': [2], 'MASK': [0]}
    else:
        map = {'DCE_mix': [1], 'MASK': [0]}
    return map[seq_id]

def sort_studies_by_date(sample_dir):
    studies_names = os.listdir(sample_dir)
    delta = [ser_name[0:5] for ser_name in studies_names]
    sorted_inx = np.argsort(delta)
    studies_names_sorted = [studies_names[inx] for inx in sorted_inx]
    if len(sorted_inx) == 2:
        studies_names_sorted.append('NAN')
    if len(sorted_inx) == 1:
        studies_names_sorted.extend(['NAN'] * 2)
    return studies_names_sorted

def create_lesion_prop_list_from_annotation_slices_range(root_path=None,lesion_table_file=None):
    if root_path is None:
        root_path = '/projects/msieve2/Platform/BigMedilytics/Data/Duke-Breast-Cancer-MRI/manifest-1607053360376/'
    if lesion_table_file is None:
        lesion_table_file = '/gpfs/haifa/projects/m/msieve_dev3/usr/Tal/my_research/CURIE/DUKE/experiments/new.csv'

    static_pipeline = DukeLesionProp.static_pipeline(root_path=root_path, select_series_func=duke.get_selected_series_index)
    dynamic_pipeline = DukeLesionProp.dynamic_pipeline()
    patient_dict = {}
    cols_list = [
            'centroid_T0', 'bbox_T0', 'volume_T0', 'elongation_T0', 'size_bbox_x_T0', 'size_bbox_y_T0', 'size_bbox_z_T0',
            'ser_name_T0',
            'centroid_T1', 'bbox_T1', 'volume_T1', 'elongation_T1', 'size_bbox_x_T1', 'size_bbox_y_T1', 'size_bbox_z_T1',
            'ser_name_T1',
            'centroid_T2', 'bbox_T2', 'volume_T2', 'elongation_T2', 'size_bbox_x_T2', 'size_bbox_y_T2', 'size_bbox_z_T2',
            'ser_name_T2',
        ]
    sample_ids = DukeLesionProp.sample_ids()#[:2]
    for sample_id in tqdm(sample_ids):
            print(sample_id)
            try:
                sample_dir = root_path +'Duke-Breast-Cancer-MRI/'+ sample_id
                studies_list = sort_studies_by_date(sample_dir)

                cols_list = []
                lesion_prop_list = ()

                for study_inx, study in enumerate(studies_list):
                    try:
                        sample_dict = create_initial_sample(sample_id)
                        sample_dict = static_pipeline(sample_dict)
                        sample_dict = dynamic_pipeline(sample_dict)
                        ser = [study]
                        lesion_prop = sample_dict['data.lesion_prop']
                        cols =  sample_dict['data.lesion_prop_col']
                    except:
                        ser = None
                        cols = ['centroid', 'bbox', 'volume', 'elongation', 'size_bbox_x', 'size_bbox_y', 'size_bbox_z']
                        lesion_prop = [(None,
                                                None,
                                                None,
                                                None,
                                                None,
                                                None,
                                                None,
                                                )]
                        cols = cols + ['ser_name']
                    cols_list = cols_list + [col + '_T' + str(study_inx) for col in cols]
                    lesion_prop_list = lesion_prop_list + lesion_prop[0] + tuple([ser])
                    patient_dict[sample_id] = lesion_prop_list

            except:
                print('No MRI')
                lesion_prop = [(None,
                                None,
                                None,
                                None,
                                None,
                                None,
                                None,
                                None,
                                None,
                                None,
                                None,
                                None,
                                None,
                                None,
                                None,
                                None,
                                None,
                                None,
                                None,
                                None,
                                None,
                                None,
                                None,
                                None,
                                )]
                lesion_prop_list = lesion_prop[0]
                patient_dict[sample_id] = lesion_prop_list[0]

    df_lesions = pd.DataFrame.from_dict(patient_dict, orient='index')
    df_lesions.columns = cols_list
    df_lesions.index.names = ['Patient ID DICOM']
    df_lesions.to_csv(lesion_table_file)


class DukeLesionProp(Duke):

    @staticmethod
    def sample_ids():
        annotations_df = get_duke_raw_annotations_df()
        return annotations_df['Patient ID'].values
    @staticmethod
    def dynamic_pipeline():

        def get_annotations(sample_id):
            patient_annotations_df = annotations_df[annotations_df['Patient ID'] == sample_id]
            return patient_annotations_df

        annotations_df = get_duke_raw_annotations_df()

        steps = [(ops_mri.OpExtractLesionPropFromBBoxAnotation(get_annotations), dict(key_in_ref_volume='data.input.ref_volume',
                                                                                      key_out_lesion_prop='data.lesion_prop',
                                                                                      key_out_cols='data.lesion_prop_col'))]

        dynamic_pipeline = PipelineDefault("dynamic", steps)

        return dynamic_pipeline

class DukeGetRadiomics(Duke):

    @staticmethod
    def sample_ids():
        annotations_df = get_duke_raw_annotations_df()
        return annotations_df['Patient ID'].values
    @staticmethod
    def dynamic_pipeline():

        def get_annotations(sample_id):
            patient_annotations_df = annotations_df[annotations_df['Patient ID'] == sample_id]
            return patient_annotations_df

        annotations_df = get_duke_raw_annotations_df()

        steps = [(ops_mri.OpExtractLesionPropFromBBoxAnotation(get_annotations), dict(key_in_ref_volume='data.input.ref_volume',
                                                                                      key_out_lesion_prop='data.lesion_prop',
                                                                                      key_out_cols='data.lesion_prop_col'))]

        dynamic_pipeline = PipelineDefault("dynamic", steps)

        return dynamic_pipeline

if __name__ == "__main__":
    create_lesion_prop_list_from_annotation_slices_range()




metadata_path =  '/gpfs/haifa/projects/m/msieve2/Platform/BigMedilytics/Data/ISPY2/manifest-Training-Set/metadata.csv'
# imaging paths
root_path = os.path.join('/gpfs/haifa/projects/m/msieve_dev3/usr/Tal/fus_sessions/', 'ISPY2/V6_100X100X9_tight_dures/')
path2cache = root_path+'/cache_03DWI+4ADC+5-7DCE+8SER+9PE2+10PE5+11ROI+12MASK_not_norm/'
# path2save = '/gpfs/haifa/projects/m/msieve2/Platform/BigMedilytics/Data/ISPY2/Analytics/ImagingFeatures/radiomics_training+testing_extand.csv'
path2save = '/gpfs/haifa/projects/m/msieve2/Platform/BigMedilytics/Data/ISPY2/Analytics/ImagingFeatures/radiomics_train_FullImageNorm_oullier2.csv'
path2data = '/gpfs/haifa/projects/m/msieve2/Platform/BigMedilytics/Data/ISPY2/manifest-Training-Set/'
mask_inx = -1
# seq_vec = ['DCE0','DCE1','DCE2','DCE3','DCE4','DCE5','DCE6','DCE7']
# seq_inx_vec = [5,6,7,8,9,10,11,12]
# seq_vec = ['b0','b1','b2','b3','ADC','DCE1','DCE2','DCE3']
# seq_inx_vec = [0,1,2,3,4,5,6,7]
# seq_vec = ['ADC','DCE2']
# seq_inx_vec = [4,6]
seq_vec = ['b0','b1','b2','b3','ADC','DCE1','DCE2','DCE3','SER','PE2','PE56']
seq_inx_vec = [0,1,2,3,4,5,6,7,8,9,10]
with open(path2cache+'cache_index.pkl', 'rb') as fp:
     samples_list = pkl.load(fp)

# paramsFile = os.path.abspath(os.path.join('.', 'params_radiomics.yaml'))

setting= {}
setting['normalize']= True
setting['normalizeScale']= 100
setting['binWidth']= 5
setting['preCrop'] = True

# Instantiate the extractor
extractor = featureextractor.RadiomicsFeatureExtractor(**setting)

print('Extraction parameters:\n\t', extractor.settings)
print('Enabled filters:\n\t', extractor.enabledImagetypes)
print('Enabled features:\n\t', extractor.enabledFeatures)
applyLog = False
applyWavelet = False
samples_dict = {}
applyNormalizationInTumorROI =False
tumorArea = False
breastArea = False
for inx,key in enumerate(samples_list.keys()):
        if key!=('ACRIN-6698-618860', 0):#('ACRIN-6698-618860',0):# or key==('ACRIN-6698-771443', 0) or key==('ACRIN-6698-342959',0) or key==('ACRIN-6698-511851',0):
            continue

    # if len(samples_list.keys())-4<inx:
        with gzip.open(path2cache + samples_list[key], 'rb') as pickle_file:
            sample = pkl.load(pickle_file)

        print(sample['data']['patient_num'])
        samples_dict[sample['data']['patient_num']] = {}
        for mask_type in ['full','edge']:
            for T in [0,1,2]:
                        vol = sample['data']['input_full_vol_T'+str(T)]
                        vol_np = vol.cpu().detach().numpy()
                        maskPath = sitk.GetImageFromArray(vol_np[mask_inx,:,:,:])
                        maskPath_binary = sitk.Cast(maskPath,sitk.sitkInt8)
                        maskPath_edge = sitk.BinaryDilate(maskPath_binary) - sitk.BinaryErode(maskPath_binary)
                        if mask_type=='edge':
                            maskPath = maskPath_edge
                        if applyNormalizationInTumorROI:
                            if tumorArea:
                                dil_filter = sitk.BinaryDilateImageFilter()
                                dil_filter.SetKernelRadius(20)
                                tumor_extand_mask = dil_filter.Execute(maskPath_binary)
                            if breastArea:
                                vol_shape = vol_np.shape

                                image = vol_np[6,int(vol_shape[1]/2),:,:]
                                image_norm = (image - image.min()) / (image.max() - image.min()) * 256
                                image_rgb = image_norm.astype(np.uint8)
                                image_seg = FCM(image=image_rgb, image_bit=8, n_clusters=2, m=2, epsilon=0.05,
                                                max_iter=100)
                                # image_seg = EnFCM(image=image_rgb, image_bit=8, n_clusters=4, m=2, epsilon=0.05, max_iter=100,neighbour_effect=8,kernel_size=4)
                                image_seg.form_clusters()
                                image_seg_res = image_seg.segmentImage()
                                breast_label = 1 -image_seg_res[2,2]
                                mask_breast = np.zeros(image_seg_res.shape)
                                mask_breast[image_seg_res==breast_label]=1
                                a = 1
                                # delta_crop= 30
                                # mask_breast_croppe
                                # buttom_mask = mask_breast[np.min(np.where(mask_breast==1)[0])::,:]
                                # top_mask = mask_breast[0:np.min(np.where(mask_breast==1)[0]),:]
                                # if sum(sum(buttom_mask)):
                                #     mask_breast[((np.max(np.where(mask_breast == 1)[0]))-delta_crop)::,:] = 0
                                # else:
                                #     mask_breast[0:((np.min(np.where(mask_breast == 1)[0])) + delta_crop), :] = 0
                                #
                                # a=1
                                if 0:
                                    thresh_img = sitk.GetImageFromArray(image_seg_res.astype(np.uint8))
                                    cleaned_thresh_img = sitk.BinaryOpeningByReconstruction(thresh_img, [2, 2, 2])
                                    cleaned_thresh_img = sitk.BinaryClosingByReconstruction(cleaned_thresh_img, [2, 2, 2])
                                    dist_img = sitk.SignedMaurerDistanceMap(cleaned_thresh_img != 0, insideIsPositive=False,
                                                                            squaredDistance=False, useImageSpacing=False)
                                    radius = 1
                                    # Seeds have a distance of "radius" or more to the object boundary, they are uniquely labelled.
                                    seeds = sitk.ConnectedComponent(dist_img < -radius)
                                    # Relabel the seed objects using consecutive object labels while removing all objects with less than 15 pixels.
                                    seeds = sitk.RelabelComponent(seeds, minimumObjectSize=15)
                                    # Run the watershed segmentation using the distance map and seeds.
                                    ws = sitk.MorphologicalWatershedFromMarkers(dist_img, seeds, markWatershedLine=True)
                                    ws = sitk.Mask(ws, sitk.Cast(cleaned_thresh_img, ws.GetPixelID()))
                                    mask_vol = sitk.GetArrayFromImage(ws)
                                    mask_breast = np.zeros(mask_vol.shape)
                                    mask_breast[mask_vol==breast_label]=1


                        for seq_inx,seq in zip(seq_inx_vec,seq_vec):
                            if T==2 and seq=='PE56':
                                continue
                            print(seq)
                            print(T)
                            print(mask_type)
                            imagePath = sitk.GetImageFromArray(vol_np[seq_inx, :, :, :])
                            if applyNormalizationInTumorROI:
                                if tumorArea:
                                    firstorder_tmp = radiomics.firstorder.RadiomicsFirstOrder(imagePath,tumor_extand_mask,**setting)
                                    firstorder_tmp.enableFeatureByName('Mean',True)
                                    firstorder_tmp.enableFeatureByName('Variance', True)
                                    results_tmp= firstorder_tmp.execute()
                                    print(results_tmp)
                                    imagePath = (imagePath-results_tmp['Mean'])/np.sqrt(results_tmp['Variance'])
                                if breastArea:
                                    vol_slice = vol_np[seq_inx, int(vol_shape[1]/2), :, :]
                                    img_mean = np.mean(vol_slice[mask_breast==1])
                                    img_std = np.std(vol_slice[mask_breast==1])
                                    imagePath = (imagePath-img_mean)/img_std

                            result = extractor.execute(imagePath, maskPath)
                            keys_ = list(result.keys())
                            for key in keys_:
                                new_key = key+'seq'+seq+'_T'+str(T)+'_'+mask_type
                                result[new_key] = result.pop(key)
                            samples_dict[sample['data']['patient_num']].update(result)

                            if applyLog:
                              sigmaValues = np.arange(5., 0., -.5)[::1]
                              for logImage, imageTypeName, inputKwargs in imageoperations.getLoGImage(imagePath, maskPath,
                                                                                                      sigma=sigmaValues):
                                logFirstorderFeatures = firstorder.RadiomicsFirstOrder(logImage, maskPath, **inputKwargs)
                                logFirstorderFeatures.enableAllFeatures()
                                result = logFirstorderFeatures.execute()
                                keys_ = list(result.keys())
                                for key in keys_:
                                  new_key = key + 'seq' + seq + '_T' + str(T)+'_'+mask_type
                                  result[new_key] = result.pop(key)
                                samples_dict[sample['data']['patient_num']].update(result)
                            #
                            # Show FirstOrder features, calculated on a wavelet filtered image
                            #
                            if applyWavelet:
                              for decompositionImage, decompositionName, inputKwargs in imageoperations.getWaveletImage(imagePath,
                                                                                                                        maskPath):
                                waveletFirstOrderFeaturs = firstorder.RadiomicsFirstOrder(decompositionImage, maskPath, **inputKwargs)
                                waveletFirstOrderFeaturs.enableAllFeatures()
                                result = waveletFirstOrderFeaturs.execute()
                                keys_ = list(result.keys())
                                for key in keys_:
                                  new_key = key + 'seq' + seq + '_T' + str(T)+'_'+mask_type
                                  result[new_key] = result.pop(key)
                                samples_dict[sample['data']['patient_num']].update(result)

pd.DataFrame(samples_dict).T.to_csv(path2save)