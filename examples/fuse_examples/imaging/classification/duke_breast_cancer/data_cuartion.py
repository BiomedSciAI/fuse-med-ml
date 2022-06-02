import os
import pandas as pd
import numpy as np
import SimpleITK as sitk
from datetime import datetime,timedelta
from fuse.utils import NDict

from fuseimg.datasets.duke import Duke
from fuse.data import DatasetDefault
from fuseimg.data.ops.ops_mri import OpExtractDicomsPerSeq, OpLoadDicomAsStkVol, OpGroupDCESequences, OpSelectVolumes, OpResampleStkVolsBasedRef
from fuseimg.datasets.duke import _get_sequence_2_series_desc_mapping,OpDukeSampleIDDecode

def get_selected_series_index(sample_id, seq_id):
    patient_id = sample_id[0]
    if patient_id in ['Breast_MRI_120', 'Breast_MRI_596']:
        map = {'DCE_mix': [2], 'MASK': [0]}
    else:
        map = {'DCE_mix': [1], 'MASK': [0]}
    return map[seq_id]

def get_zeros_vol(vol):
    if vol.GetNumberOfComponentsPerPixel() >1:
        ref_zeros_vol = sitk.VectorIndexSelectionCast(vol,0)
    else:
        ref_zeros_vol = vol
    zeros_vol = np.zeros_like(sitk.GetArrayFromImage(ref_zeros_vol))
    zeros_vol = sitk.GetImageFromArray(zeros_vol)
    zeros_vol.CopyInformation(ref_zeros_vol)
    return zeros_vol


def extarct_lesion_prop_from_mask(mask):
    ma_centroid = mask > 0.5

    dist_img = sitk.SignedMaurerDistanceMap(ma_centroid,insideIsPositive=False,squaredDistance=False,useImageSpacing=False)
    seeds = sitk.ConnectedComponent(dist_img<40)
    seeds = sitk.RelabelComponent(seeds,minimumObjectSize=3)
    ws = sitk.MorphologicalWatershedFromMarkers(dist_img,seeds,markWatershedLine=True)
    ws = sitk.Mask(ws,sitk.Cast(ma_centroid,ws.GetPixelID()))

    shape_stats = sitk.LabelShapeStatisticsImageFilter()
    shape_stats.ComputeOrientedBoundingBoxOn()
    shape_stats.Execute(ws)
    stats_list = [(shape_stats.GetCentroid(i),
                   shape_stats.GetBoundingBox(i),
                   shape_stats.GetPhysicalSize(i),
                   shape_stats.GetElongation(i),
                   shape_stats.GetOrientedBoundingBoxSize(i)[0],
                   shape_stats.GetOrientedBoundingBoxSize(i)[1],
                   shape_stats.GetOrientedBoundingBoxSize(i)[2],
                   )
                  for i in shape_stats.GetLabels()]

    cols = ["centroid","bbox","volume","elongation","size_bbox_x","size_bbox_y","size_bbox_z"]
    return stats_list,cols

def extarct_lesion_prop_from_annotation(vol_ref,bbox_coords,start_slice,end_slice):
    mask = get_zeros_vol(vol_ref)
    mask_np = sitk.GetArrayFromImage(mask)
    mask_np[start_slice:end_slice,bbox_coords[0][1]:bbox_coords[1][1],bbox_coords[0][0]:bbox_coords[1][0]] = 1.0
    mask_final = sitk.GetImageFromArray(mask_np)
    mask_final.CopyInformation(vol_ref)
    mask_final = sitk.Image(mask_final)
    return extarct_lesion_prop_from_mask(mask_final)


def sort_studies_by_date(patient_dir):
    ser_names = os.listdir(path_to_dataset + patient_id)
    # delta = [(datetime.strptime(ser_name[0:10],'%m-%d-%Y')-ref_date).days for ser_name in ser_names]
    delta = [ser_name[0:5] for ser_name in ser_names]
    study_names = [ser_name[0:5] for ser_name in ser_names]
    sorted_inx = np.argsort(delta)
    cols_list = []
    lesion_prop_list = ()
    if len(sorted_inx) == 2:
        sorted_inx = np.append(sorted_inx, np.array([2]), axis=0)
    if len(sorted_inx) == 1:
        sorted_inx = np.append(sorted_inx, np.array([1, 2]), axis=0)
    sorted_inx = sorted_inx[0:3]


def mri_processor(sample_dict):
    root_path = '/projects/msieve2/Platform/BigMedilytics/Data/Duke-Breast-Cancer-MRI/manifest-1607053360376/'
    data_path = os.path.join(root_path, 'Duke-Breast-Cancer-MRI')
    metadata_path = os.path.join(root_path, 'metadata.csv')

    SEQ_IDS = ['DCE_mix_ph1', 'DCE_mix_ph3']

    seq_dict = _get_sequence_2_series_desc_mapping(metadata_path)


    # step 1: map sample_ids to
    op = OpDukeSampleIDDecode(data_path=data_path)
    sample_dict = op(sample_dict=sample_dict, key_out='data.input.mri_path', op_id=1)

    # step 2: read files info for the sequences
    op = OpExtractDicomsPerSeq(seq_ids=SEQ_IDS, seq_dict=seq_dict, use_order_indicator=False)
    sample_dict = op(sample_dict,
                     key_in='data.input.mri_path',
                     key_out_sequences='data.input.sequence_ids',
                     key_out_path_prefix='data.input.path.',
                     key_out_dicoms_prefix='data.input.dicoms.',
                     key_out_series_num_prefix='data.input.series_num.'
                     )

    # step 3: Load STK volumes of MRI sequences
    op = OpLoadDicomAsStkVol(reverse_order=False, is_file=False)
    sample_dict = op(sample_dict,
                     key_in_seq_ids='data.input.sequence_ids',
                     key_in_path_prefix='data.input.path.',
                     key_in_dicoms_prefix='data.input.dicoms.',
                     key_out_prefix='data.input.volumes.')

    # step 4: group DCE sequnces into DCE_mix
    op = OpGroupDCESequences()
    sample_dict = op(sample_dict,
                     key_sequence_ids='data.input.sequence_ids',
                     key_path_prefix='data.input.path.',
                     key_series_num_prefix='data.input.series_num.',
                     key_volumes_prefix='data.input.volumes.',
                     )

    # step 5: select single volume from DCE_mix sequence
    op = OpSelectVolumes(subseq_to_use=['DCE_mix'], get_indexes_func=get_selected_series_index)
    sample_dict = op(sample_dict,
                     key_in_path_prefix='data.input.path.',
                     key_in_volumes_prefix='data.input.volumes.',
                     key_out_path_prefix='data.input.selected_path.',
                     key_out_volumes_prefix='data.input.selected_volumes.')

    op_resample_stk_vols_based_ref = OpResampleStkVolsBasedRef(reference_inx=0, interpolation='bspline')
    sample_dict = op_resample_stk_vols_based_ref(sample_dict,
                                                 key_seq_ids='data.input.sequence_ids',
                                                 key_seq_volumes_prefix='data.input.selected_volumes.',
                                                 key_out_prefix='data.input.selected_volumes_resampled.',
                                                 )
    return sample_dict



method = 'create_lesion_prop_list_from_annotation_slices_range'
if method == 'create_lesion_prop_list_from_annotation_slices_range':
    DATADIR = '/projects/msieve2/Platform/BigMedilytics/Data/Duke-Breast-Cancer-MRI/manifest-1607053360376/Duke-Breast-Cancer-MRI/'
    DATATABLE = '/gpfs/haifa/projects/m/msieve2/Platform/BigMedilytics/Data/Duke-Breast-Cancer-MRI/manifest-1607053360376/metadata.csv'
    DBDIR = '/gpfs/haifa/projects/m/msieve2/Platform/BigMedilytics/Data/Duke-Breast-Cancer-MRI/manifest-1607053360376/metadata.csv'
    ANNOTATION = '/gpfs/haifa/projects/m/msieve2/Platform/BigMedilytics/Data/Duke-Breast-Cancer-MRI/Annotation_Boxes.csv'
    PATH2SAVE = '/gpfs/haifa/projects/m/msieve_dev3/usr/Tal/my_research/CURIE/DUKE/experiments/'

    DATATABLEANDLESION_0 = PATH2SAVE + '/lesion_info_30092021.csv'
    DATATABLEANDLESION_1 = '/gpfs/haifa/projects/m/msieve2/Platform/BigMedilytics/Data/Duke-Breast-Cancer-MRI/Analytics/Clinical/revision_20211010/Clinical_and_Other_Features_processed.csv'

    SEED_NUMBER = 1
    DB_VER = '10112021_PCRsubset'
    method = 'cross_val_pcr'
    EXLUDE = []

    l_seq = pd.read_csv(DBDIR)
    l_info = pd.read_csv(DATATABLE)
    l_annotation = pd.read_csv(ANNOTATION)
    seq_to_use_full = list(l_seq['Series Description'].value_counts().keys())

    SER_INX_TO_USE = {}
    SER_INX_TO_USE['all'] = {'DCE_mix': [0]}

    opt_seq = [
        'dyn',

    ]
    my_keys = ['DCE_mix']

    seq_to_use = {}
    for opt_seq_tmp, my_key in zip(opt_seq, my_keys):
        tt = [s for s in seq_to_use_full if opt_seq_tmp in s]

        for tmp in tt:
            seq_to_use[tmp] = my_key

    path_to_dataset = DATADIR
    Ktrain_data_path = ''
    sample_ids =  list(l_seq['Subject ID'].unique())
    # mri_processor = FuseDicomMRIProcessor(seq_dict=seq_to_use, seq_to_use=['DCE_mix'], subseq_to_use=['DCE_mix'],
    #                                       ser_inx_to_use=SER_INX_TO_USE, reference_inx=0, use_order_indicator=False)
    ref_date = datetime(1000, 1, 1)
    cols_list = [
        'centroid_T0', 'bbox_T0', 'volume_T0', 'elongation_T0', 'size_bbox_x_T0', 'size_bbox_y_T0', 'size_bbox_z_T0',
        'ser_name_T0',
        'centroid_T1', 'bbox_T1', 'volume_T1', 'elongation_T1', 'size_bbox_x_T1', 'size_bbox_y_T1', 'size_bbox_z_T1',
        'ser_name_T1',
        'centroid_T2', 'bbox_T2', 'volume_T2', 'elongation_T2', 'size_bbox_x_T2', 'size_bbox_y_T2', 'size_bbox_z_T2',
        'ser_name_T2',
    ]
    lesion_prop_list_patient = []

    patient_list = []
    patient_dict = {}
    patient_list = list(l_seq['Subject ID'].unique())

    for pat_inx, pat in enumerate(patient_list):
        patient_id = pat
        print(patient_id)
        print(str(pat_inx / len(patient_list)))
        try:
            ser_names = os.listdir(path_to_dataset + patient_id)
            # delta = [(datetime.strptime(ser_name[0:10],'%m-%d-%Y')-ref_date).days for ser_name in ser_names]
            delta = [ser_name[0:5] for ser_name in ser_names]
            study_names = [ser_name[0:5] for ser_name in ser_names]
            sorted_inx = np.argsort(delta)
            cols_list = []
            lesion_prop_list = ()
            if len(sorted_inx) == 2:
                sorted_inx = np.append(sorted_inx, np.array([2]), axis=0)
            if len(sorted_inx) == 1:
                sorted_inx = np.append(sorted_inx, np.array([1, 2]), axis=0)
            sorted_inx = sorted_inx[0:3]

            for inx, ser_inx in enumerate(sorted_inx):
                try:
                    TEST_PATIENT_ID, TEST_STUDY_ID = 'Breast_MRI_900', '01-01-1990-BREASTROUTINE DYNAMICS-51487'
                    sample_dict = NDict({'data': {'sample_id': (TEST_PATIENT_ID, TEST_STUDY_ID)}})

                    prostate_data_path = os.path.join(path_to_dataset, patient_id + '/' + ser_names[ser_inx] + '/')
                    # vol_4D, vol_ref = mri_processor.__call__((prostate_data_path, Ktrain_data_path, patient_id))
                    vol_4D, vol_ref = mri_processor.__call__((prostate_data_path, Ktrain_data_path, patient_id))

                    zsize = vol_ref.GetSize()[2]
                    bbox_coords = ((l_annotation[l_annotation['Patient ID'] == patient_id]['Start Column'].values[0],
                                    l_annotation[l_annotation['Patient ID'] == patient_id]['Start Row'].values[0]),
                                   (l_annotation[l_annotation['Patient ID'] == patient_id]['End Column'].values[0],
                                    l_annotation[l_annotation['Patient ID'] == patient_id]['End Row'].values[0]))

                    start_slice = l_annotation[l_annotation['Patient ID'] == patient_id]['Start Slice'].values[0]
                    end_slice = l_annotation[l_annotation['Patient ID'] == patient_id]['End Slice'].values[0]
                    lesion_prop, cols = extarct_lesion_prop_from_annotation(vol_ref, bbox_coords, start_slice,
                                                                            end_slice)
                    ser = [ser_names[ser_inx]]
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
                cols_list = cols_list + [col + '_T' + str(inx) for col in cols]
                lesion_prop_list = lesion_prop_list + lesion_prop[0] + tuple([ser])
            patient_dict[patient_id] = lesion_prop_list
        except:
            print('No MRI')
            cols_list = [
                'centroid_T0', 'bbox_T0', 'volume_T0', 'elongation_T0', 'size_bbox_x_T0', 'size_bbox_y_T0',
                'size_bbox_z_T0', 'ser_name_T0',
                'centroid_T1', 'bbox_T1', 'volume_T1', 'elongation_T1', 'size_bbox_x_T1', 'size_bbox_y_T1',
                'size_bbox_z_T1',
                'ser_name_T1',
                'centroid_T2', 'bbox_T2', 'volume_T2', 'elongation_T2', 'size_bbox_x_T2', 'size_bbox_y_T2',
                'size_bbox_z_T2',
                'ser_name_T2',
            ]
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
            patient_dict[pat] = lesion_prop_list[0]

        df_lesions = pd.DataFrame.from_dict(patient_dict, orient='index')
        tt = df_lesions.to_csv(DATATABLEANDLESION_0)
    # tt = pd.concat([l_info,df_lesions],axis=1).to_csv(DATATABLEANDLESION_0)
