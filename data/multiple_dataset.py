import tensorflow as tf
from torch.utils.data import Dataset

from data.data_loader import build_dataset, tf_to_torch


class CombinedDataset(Dataset):
    def __init__(self, time_sequence_length=6):
        trajectory_dataset_list = []
        dataset_trajectory_transform_list = []
        builder_dir_list = ['gs://gresearch/robotics/toto/0.1.0', 'gs://gresearch/robotics/bridge/0.1.0']
        dataset_name_list = ['toto', 'bridge']

        for idx, builder_dir in enumerate(builder_dir_list):
            print('start loading', builder_dir)
            trajectory_dataset, dataset_trajectory_transform = build_dataset(
                dataset_name=dataset_name_list[idx], builder_dir=builder_dir, trajectory_length=time_sequence_length)
            trajectory_dataset_list.append(trajectory_dataset)
            dataset_trajectory_transform_list.append(dataset_trajectory_transform)

        template_dataset_trajectory_transform = dataset_trajectory_transform_list[0]
        for dataset_trajectory_transform in dataset_trajectory_transform_list:
            assert dataset_trajectory_transform.expected_tensor_spec == template_dataset_trajectory_transform.expected_tensor_spec

        combined_dataset = tf.data.Dataset.sample_from_datasets(trajectory_dataset_list)
        # combined_dataset = combined_dataset.batch(2)
        self.combined_dataset_it = iter(combined_dataset)

    def __len__(self):
        return int(1e8)

    def __getitem__(self, idx):
        example = next(self.combined_dataset_it)
        for key, value in example.items():
            if isinstance(value, dict):
                for subk, subv in value.items():
                    example[key][subk] = tf_to_torch(subv)
            else:
                example[key] = tf_to_torch(value)

        return example
