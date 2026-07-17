import os
from OptimizedDataGenerator_v3 import *

def generate_tfrecords(
    dataset_dir, 
    train_batch_size, 
    val_batch_size,
    to_standardize=False,
    custom_standardization=False,
    dataset_mean=None,
    dataset_std=None,
    dataset_max=None,
    dataset_min=None,
    norm_factor_pos=None,
    norm_factor_neg=None,
    log_compression=False, 
    select_contained=False,
    timeslices=2, 
    labels_list=['x-midplane','y-midplane','cotAlpha','cotBeta'],
    seed=10, 
    max_workers=1, 
    tfrecords_exist=False, 
    model_type=None,
    labels_scale=None,
    test_only=False,
):
    # determine labels_list
    if 'Slim' in model_type:
        labels_list=['x-midplane','y-midplane','cotBeta']
    else:
        labels_list=['x-midplane','y-midplane','cotAlpha','cotBeta']
        
    # determine the time stamps to use
    if timeslices==2:
        time_stamps=[0,19]
    elif timeslices==20:
        time_stamps=-1
    
    # determine selections
    contained_label=''
    if select_contained:
        contained_label='_contained'
        select_contained=True
    slim_label=''
    if 'Slim' in model_type:
        slim_label='_slim'
    standardize_label=''
    if to_standardize:
        standardize_label='_std'
    log_compression_label=''
    if log_compression:
        log_compression_label='_log'
        
    # determine input directories of parquets and output directories of tfrecords

    # FOR PRODUCING TEST SET ONLY
    if test_only:
        dataset_test_dir=os.path.join(dataset_dir, f"test{contained_label}")
        tfrecords_dir=os.path.join(dataset_dir, "TFR_files", f"{timeslices}t")
        tfrecords_dir_test=os.path.join(tfrecords_dir, f"TFR_test{contained_label}{slim_label}{standardize_label}{log_compression_label}")
        os.makedirs(tfrecords_dir_test, exist_ok=True)
        if tfrecords_exist:
            generator=OptimizedDataGenerator(
                dataset_base_dir=dataset_test_dir,
                file_type="parquet",
                data_format="3D",
                batch_size=val_batch_size,
                file_count=len(os.listdir(dataset_test_dir)),
                to_standardize=to_standardize,
                log_compression=log_compression,
                select_contained=select_contained,
                include_y_local=False,
                labels_list=labels_list,
                input_shape=(timeslices,16,16),
                transpose=(0,2,3,1),
                shuffle=False, 
                files_from_end=True,
                seed=seed,
                labels_scale=labels_scale,
    
                tfrecords_dir=tfrecords_dir_test,
                use_time_stamps=time_stamps,
                max_workers=max_workers,
                load_from_tfrecords_dir=tfrecords_dir_test,
            )
        else:
            generator=OptimizedDataGenerator(
                dataset_base_dir=dataset_test_dir,
                file_type="parquet",
                data_format="3D",
                batch_size=val_batch_size,
                file_count=len(os.listdir(dataset_test_dir)),
                to_standardize=to_standardize,
                log_compression=log_compression,
                select_contained=select_contained,
                include_y_local=False,
                labels_list=labels_list,
                input_shape=(timeslices,16,16),
                transpose=(0,2,3,1),
                shuffle=False, 
                files_from_end=True,
                seed=seed,
                labels_scale=labels_scale,

                custom_standardization=custom_standardization,
                dataset_mean=dataset_mean,
                dataset_std=dataset_std,
                dataset_max=dataset_max,
                dataset_min=dataset_min,
                norm_factor_pos=norm_factor_pos,
                norm_factor_neg=norm_factor_neg,
                
                tfrecords_dir=tfrecords_dir_test,
                use_time_stamps=time_stamps,
                max_workers=max_workers,
            )
        return dataset_test_dir, tfrecords_dir_test
        
    else:
        dataset_train_dir=os.path.join(dataset_dir, f"train{contained_label}")
        dataset_validation_dir=os.path.join(dataset_dir, f"test{contained_label}")
    
        tfrecords_dir=os.path.join(dataset_dir, "TFR_files", f"{timeslices}t")
        tfrecords_dir_train=os.path.join(tfrecords_dir, f"TFR_train{contained_label}{slim_label}{standardize_label}{log_compression_label}")
        tfrecords_dir_val=os.path.join(tfrecords_dir, f"TFR_test{contained_label}{slim_label}{standardize_label}{log_compression_label}")
    
    
        dirs_to_create=[
        tfrecords_dir_train,
        tfrecords_dir_val,
        ]
        for directory in dirs_to_create:
            os.makedirs(directory, exist_ok=True)
    
        if tfrecords_exist:
            training_generator=OptimizedDataGenerator(
                dataset_base_dir=dataset_train_dir,
                file_type="parquet",
                data_format="3D",
                batch_size=train_batch_size,
                file_count=len(os.listdir(dataset_train_dir)),
                to_standardize=to_standardize, 
                log_compression=log_compression, 
                select_contained=select_contained,
                include_y_local=False,
                labels_list=labels_list,
                input_shape=(timeslices,16,16),
                transpose=(0,2,3,1),
                shuffle=False,
                seed=seed,
                labels_scale=labels_scale,
    
                tfrecords_dir=tfrecords_dir_train,
                use_time_stamps=time_stamps,
                max_workers=max_workers,
                load_from_tfrecords_dir=tfrecords_dir_train,
            )
    
            validation_generator=OptimizedDataGenerator(
                dataset_base_dir=dataset_validation_dir,
                file_type="parquet",
                data_format="3D",
                batch_size=val_batch_size,
                file_count=len(os.listdir(dataset_validation_dir)),
                to_standardize=to_standardize,
                log_compression=log_compression,
                select_contained=select_contained,
                include_y_local=False,
                labels_list=labels_list,
                input_shape=(timeslices,16,16),
                transpose=(0,2,3,1),
                shuffle=False, 
                files_from_end=True,
                seed=seed,
                labels_scale=labels_scale,
    
                tfrecords_dir=tfrecords_dir_val,
                use_time_stamps=time_stamps,
                max_workers=max_workers,
                load_from_tfrecords_dir=tfrecords_dir_val,
            )
        
        else:
            training_generator=OptimizedDataGenerator(
                dataset_base_dir=dataset_train_dir,
                file_type="parquet",
                data_format="3D",
                batch_size=train_batch_size,
                file_count=len(os.listdir(dataset_train_dir)),
                to_standardize=to_standardize, 
                log_compression=log_compression, 
                select_contained=select_contained,
                include_y_local=False,
                labels_list=labels_list,
                input_shape=(timeslices,16,16),
                transpose=(0,2,3,1),
                shuffle=False,
                seed=seed, 
                labels_scale=labels_scale,

                custom_standardization=custom_standardization,
                dataset_mean=dataset_mean,
                dataset_std=dataset_std,
                dataset_max=dataset_max,
                dataset_min=dataset_min,
                norm_factor_pos=norm_factor_pos,
                norm_factor_neg=norm_factor_neg,
    
                tfrecords_dir=tfrecords_dir_train,
                use_time_stamps=time_stamps,
                max_workers=max_workers,
            )
            
            validation_generator=OptimizedDataGenerator(
                dataset_base_dir=dataset_validation_dir,
                file_type="parquet",
                data_format="3D",
                batch_size=val_batch_size,
                file_count=len(os.listdir(dataset_validation_dir)),
                to_standardize=to_standardize,
                log_compression=log_compression,
                select_contained=select_contained,
                include_y_local=False,
                labels_list=labels_list,
                input_shape=(timeslices,16,16),
                transpose=(0,2,3,1),
                shuffle=False, 
                files_from_end=True,
                seed=seed,
                labels_scale=labels_scale,

                custom_standardization=custom_standardization,
                dataset_mean=dataset_mean,
                dataset_std=dataset_std,
                dataset_max=dataset_max,
                dataset_min=dataset_min,
                norm_factor_pos=norm_factor_pos,
                norm_factor_neg=norm_factor_neg,
                
                tfrecords_dir=tfrecords_dir_val,
                use_time_stamps=time_stamps,
                max_workers=max_workers,
            ) 
    
        return dataset_train_dir, dataset_validation_dir, tfrecords_dir_train, tfrecords_dir_val

def load_tfrecords(
    tfrecords_dir_train=None, 
    tfrecords_dir_val=None,
    tfrecords_dir_test=None,
    seed=10,
    noise=-1, 
    quantize=False, 
    shuffle=True,
    digitize=False,
    digitize_levels=None,
    digitize_thresholds=None,
    test_only=False,
    to_standardize=False,
    log_compression=False,
):

    if test_only:
        test_generator=OptimizedDataGenerator(
            load_from_tfrecords_dir=tfrecords_dir_test,
            shuffle=shuffle,
            seed=seed,
            noise=noise,
            quantize=quantize,
            digitize=digitize,
            digitize_levels=digitize_levels,
            digitize_thresholds=digitize_thresholds,
        )
        return test_generator
        
    training_generator=OptimizedDataGenerator(
        load_from_tfrecords_dir=tfrecords_dir_train,
        shuffle=shuffle,
        seed=seed,
        noise=noise,
        quantize=quantize,
        digitize=digitize,
        digitize_levels=digitize_levels,
        digitize_thresholds=digitize_thresholds,
    )

    validation_generator=OptimizedDataGenerator(
        load_from_tfrecords_dir=tfrecords_dir_val,
        shuffle=shuffle,
        seed=seed,
        noise=noise,
        quantize=quantize,
        digitize=digitize,
        digitize_levels=digitize_levels,
        digitize_thresholds=digitize_thresholds,
    )

    return training_generator, validation_generator