import os
import pandas as pd
import numpy as np
import SimpleITK as sitk
from datetime import datetime,timedelta
from tqdm import tqdm
from fuse.utils import NDict
from fuseimg.datasets import duke
from fuse.data.utils.sample import create_initial_sample

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



method = 'create_lesion_prop_list_from_annotation_slices_range'
if method == 'create_lesion_prop_list_from_annotation_slices_range':
    DATADIR = '/projects/msieve2/Platform/BigMedilytics/Data/Duke-Breast-Cancer-MRI/manifest-1607053360376/Duke-Breast-Cancer-MRI/'
    DATATABLE = '/gpfs/haifa/projects/m/msieve2/Platform/BigMedilytics/Data/Duke-Breast-Cancer-MRI/manifest-1607053360376/metadata.csv'
    DBDIR = '/gpfs/haifa/projects/m/msieve2/Platform/BigMedilytics/Data/Duke-Breast-Cancer-MRI/manifest-1607053360376/metadata.csv'
    ANNOTATION = '/gpfs/haifa/projects/m/msieve2/Platform/BigMedilytics/Data/Duke-Breast-Cancer-MRI/Annotation_Boxes.csv'
    PATH2SAVE = '/gpfs/haifa/projects/m/msieve_dev3/usr/Tal/my_research/CURIE/DUKE/experiments/'
    root_path = '/projects/msieve2/Platform/BigMedilytics/Data/Duke-Breast-Cancer-MRI/manifest-1607053360376/'

    DATATABLEANDLESION_0 = PATH2SAVE + '/new.csv'
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

    static_pipeline = duke.Duke.static_pipeline(root_path=root_path, select_series_func=duke.get_selected_series_index)
    sample_ids = duke.Duke.sample_ids()[:5]
    for sample_id in tqdm(sample_ids):
        print(sample_id)
        try:
            sample_dir = path_to_dataset + sample_id
            studies_list = sort_studies_by_date(sample_dir)

            cols_list = []
            lesion_prop_list = ()

            for study_inx, study in enumerate(studies_list):
                try:

                    sample_dict = NDict({'data': {'sample_id': (sample_id, study)}})
                    sample_dict = create_initial_sample(sample_id)
                    sample_dict = static_pipeline(sample_dict)

                    vol_4D = sample_dict.get('data.input.volume4D')
                    vol_ref = sample_dict.get('data.input.ref_volume')

                    zsize = vol_ref.GetSize()[2]
                    bbox_coords = ((l_annotation[l_annotation['Patient ID'] == sample_id]['Start Column'].values[0],
                                    l_annotation[l_annotation['Patient ID'] == sample_id]['Start Row'].values[0]),
                                   (l_annotation[l_annotation['Patient ID'] == sample_id]['End Column'].values[0],
                                    l_annotation[l_annotation['Patient ID'] == sample_id]['End Row'].values[0]))

                    start_slice = l_annotation[l_annotation['Patient ID'] == sample_id]['Start Slice'].values[0]
                    end_slice = l_annotation[l_annotation['Patient ID'] == sample_id]['End Slice'].values[0]
                    lesion_prop, cols = extarct_lesion_prop_from_annotation(vol_ref, bbox_coords, start_slice,
                                                                            end_slice)
                    ser = [study]
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
            patient_dict[sample_id] = lesion_prop_list[0]

        df_lesions = pd.DataFrame.from_dict(patient_dict, orient='index')
        tt = df_lesions.to_csv(DATATABLEANDLESION_0)
    # tt = pd.concat([l_info,df_lesions],axis=1).to_csv(DATATABLEANDLESION_0)
