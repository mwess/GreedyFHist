==============================
Groupwise Registration Example
==============================

In this section, we show an example for groupwise registration for the commandline interface.

------------------------------
Commandline interface example.
------------------------------

When using the command line, a ``config.toml`` has to be defined containing registration options and image topology.

Input
=====

In this part of the configuration the image topology to be registered is defined. Each section consists of a ``reference_image`` which is used for registration. Optionally, a ``reference_mask`` can be supplied as well (,e.g. for focusing on a specific region of interest). If not used, the mask will be computed by GreedyFHist. Additional data that should be transformed to the fixed image space is defined in ``input.sectionX.additional_data`` as a list. By convention, the last section defined is treated as the fixed section towards which all other moving section registered.


A full example configuration might look like this:


.. code-block::

    [options]

    output_directory = 'group_out'
    path_to_greedy = ''

    [gfh_options]

    pre_sampling_factor = 'auto'
    pre_sampling_auto_factor = 2000
    do_affine_registration = true
    do_nonrigid_registration = true
    compute_reverse_nonrigid_registration = false
    temporary_directory = 'tmp'
    remove_temporary_directory = true
    disable_mask_generation = false

    [gfh_options.segmentation]
    segmentation_class = 'YoloSegOptions'
    min_area_size = 10000
    use_tv_chambolle = true
    use_clahe = false
    fill_holes = true

    [gfh_options.affine_registration_options]

    dim = 2
    resolution = [1024, 1024]
    kernel_size = 10
    cost_function = 'ncc'
    rigid_iterations = 10000
    ia = 'ia-com-init'
    iteration_pyramid = [100, 50, 10]
    n_threads = 8
    keep_affine_transform_unbounded = true

    [gfh_options.affine_registration_options.preprocessing_options]
    moving_sr = 30
    moving_sp = 25
    fixed_sr = 30
    fixed_sp = 25
    temporary_directory = 'tmp'
    remove_temporary_directory = true
    yolo_segmentation_min_size = 5000
    enable_denoising = true


    [gfh_options.nonrigid_registration_options]
    dim = 2
    resolution = [1024, 1024]
    s1 = 5.0
    s2 = 5.0
    kernel_size = 10
    cost_function = 'ncc'
    ia = 'ia-com-init'
    iteration_pyramid = [100, 100, 50, 10]
    n_threads = 8
    use_sv = false
    use_svlb = false

    [gfh_options.nonrigid_registration_options.preprocessing_options]
    moving_sr = 30
    moving_sp = 25
    fixed_sr = 30
    fixed_sp = 25
    temporary_directory = 'tmp'
    remove_temporary_directory = true
    yolo_segmentation_min_size = 5000
    enable_denoising = false

    [gfh_options.tiling_options]
    enable_tiling = false
    tiling_mode = 'simple'
    stop_condition_tile_resolution = false
    stop_condition_pyramid_counter = true
    max_pyramid_depth = 0
    tile_overlap = 0.75
    tile_size = 1024
    min_overlap = 0.1


    [input]
    [input.section1]
    [input.section1.reference_image]

    path = 'img1.ome.tiff'

    [[input.section1.additional_data]]

    path = 'ann1.ome.tiff'

    [[input.section1.additional_data]]

    path = 'ann2.geojson'
    type = 'geojson'

    [input.section1.reference_mask]

    path = 'mask1.ome.tiff'

    [input.section2]
    [input.section2.reference_image]

    path = 'img2.ome.tiff'

    [input.section3]
    [input.section3.reference_image]

    path = 'img3.ome.tiff'


``gfh_options`` and ``options`` are covered in the config documentation.


GreedyFHist's groupwise registration can be executed the following way:

.. code-block::

    greedyfhist groupwise-registration -c groupwise_config.toml



This will result in the following output structure (shortened):

.. code-block::

    group_out/
    ├── group_transforms
    │   ...
    ├── section0
    │   ├── transformations
    │   │   ├── fixed_transform
    │   │   │   ├── attributes.json
    │   │   │   └── transform.txt
    │   │   └── moving_transform
    │   │       ├── attributes.json
    │   │       └── transform.txt
    │   └── transformed_data
    │       ├── hes_mask.ome.tif
    │       ├── hes.ome.tif
    │       └── hes_ps.csv
    ├── section1
    │   ├── transformations
    │   │   ├── fixed_transform
    │   │   │   ├── attributes.json
    │   │   │   └── transform.txt
    │   │   └── moving_transform
    │   │       ├── attributes.json
    │   │       └── transform.txt
    │   └── transformed_data
    │       ├── mts_mask.ome.tif
    │       ├── mts.ome.tif
    │       └── mts_ps.csv
    └── section2
        └── transformed_data
            └── ihc.ome.tif


`sectionX` contains the registered data as well the transformation from `sectionX` to fixed section. `group_out` contains all computed
transformations. 


-------------------------------------------------------
Groupwise registration using interactive Python session
-------------------------------------------------------

An example using the interactive Python session can be found in ``examples/notebooks/groupwise.ipynb``.