import os

from examples.fuse_examples.imaging.classification.duke_breast_cancer import replace_tensors_with_numpy, compare_dicts

os.environ["DUKE_DATA_PATH"] = "/projects/msieve2/Platform/BigMedilytics/Data/Duke-Breast-Cancer-MRI"

from fuse.data.utils.sample import create_initial_sample
from fuseimg.datasets import duke
from examples.fuse_examples.imaging.classification import duke_breast_cancer


def main():
    cache_dir = os.path.join(duke_breast_cancer.get_duke_user_dir(), 'cache_dir_regression_test')

    test_duke(os.environ["DUKE_DATA_PATH"], cache_dir=cache_dir, verbose=True)


def test_duke(root_path, cache_dir, verbose=False):
    label_type = duke.DukeLabelType.STAGING_TUMOR_SIZE
    sample_ids_2_test = [f'Breast_MRI_{i:03d}' for i in list(range(900, 901))] # + [120, 596]]

    static_pipeline = duke.Duke.static_pipeline(data_dir=root_path,
                                                select_series_func=duke.get_selected_series_index,
                                                verbose=verbose)
    dynamic_pipeline = duke.Duke.dynamic_pipeline(data_dir=root_path, label_type=label_type, verbose=verbose)

    dict_static = {}
    dict_dynamic = {}
    for sample_id in sample_ids_2_test:
        sample_dict = create_initial_sample(sample_id)
        sample_dict = static_pipeline(sample_dict)
        dict_static[sample_id] = sample_dict.flatten()

        sample_dict = dynamic_pipeline(sample_dict)
        dict_dynamic[sample_id] = replace_tensors_with_numpy(sample_dict.flatten())

    for s, dict_obj in [('dynamic', dict_dynamic)]:  # ('static', dict_static),
        pipeline_cache_file = os.path.join(cache_dir, f'{s}_pipeline_output.pkl.gz')
        if os.path.exists(pipeline_cache_file):
            ref_dict_obj = duke_breast_cancer.load_object(pipeline_cache_file)
            compare_dicts(dict_obj, ref_dict_obj)
        else:
            duke_breast_cancer.save_object(dict_obj, pipeline_cache_file)
            print("wrote", pipeline_cache_file)


if __name__ == '__main__':
    main()
