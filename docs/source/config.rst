.. _topics-config:

=======================
Configuration structure
=======================

Here we explain how configuration files are defined.

Each configuration consits of up to 3 sections: ``gfh_options``, ``options``, and ``input``.

----------
GFHOptions
----------

This section is used to declare any parameters used during registration. Below is a full configuration example. Absent field in the configuration are replaced by default parameters. For a full explanation of each parameter, see `options.py <https://github.com/mwess/GreedyFHist/blob/master/greedyfhist/options/options.py>`.


.. code-block::

    [gfh_options]

    pre_sampling_factor = 'auto'
    pre_sampling_auto_factor = 2000
    do_affine_registration = true
    do_nonrigid_registration = true
    compute_reverse_nonrigid_registration = false
    temporary_directory = 'tmp'
    remove_temporary_directory = true
    disable_mask_generation = false

    # Several options for segmentation can be tried out by commenting in one of the following
    # three options.
    [gfh_options.segmentation]

    segmentation_class = 'YoloSegOptions'
    min_area_size = 10000
    use_tv_chambolle = true
    use_clahe = false
    fill_holes = true

    #[gfh_options.segmentation]
    #
    #segmentation_class = 'TissueEntropySegOptions'
    #target_resolution = 640
    #do_clahe = true
    #use_luminosity = false
    #footprint_size = 10
    #convert_to_xyz = false
    #normalize_entropy = false
    #pre_gaussian_sigma = 0.5
    #area_opening_connectivity = 1
    #area_opening_threshold = 100
    #post_gaussian_sigma = 0.5
    #with_morphological_closing = true
    #do_fill_hole = true

    #[gfh_options.segmentation]
    #
    #segmentation_class = 'LuminosityAndAreaSegOptions'
    #target_resolution = 640
    #disk_size = 1
    #with_morphological_erosion = true
    #with_morphological_closing = false
    #min_area_size = 100
    #distance_threshold = 30
    #low_intensity_rem_threshold = 25
    #with_hole_filling = true

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
    # Tiling can be enabled by setting this property to true
    enable_tiling = false
    tiling_mode = 'simple'
    stop_condition_tile_resolution = false
    stop_condition_pyramid_counter = true
    max_pyramid_depth = 0
    tile_overlap = 0.75
    tile_size = 1024
    min_overlap = 0.1
    # Comment this option in if parallelization should be used.
    #n_procs = 4

-------
Options
-------

At the moment, there are only two fields set in ``output``: ``output_directory`` defines where registration data is 
stored (defaults to `out`) and `path_to_greedy` sets the path to the directory of the ``greedy`` executable. 
If ``greedy`` is on the ``PATH`` environment variable, this option can be skipped.


-----
Input
-----

In this section, the spatial topolgy is defined. The configuration depends on the registration mode, i.e. ``pairwise`` 
and ``groupwise`` registrations require slightly different setups. However, defining a spatial 
object is the same throughout both modes.

Image data
==========


.. code-block::

    ...
    [input.reference_image]
    path = 'image.ome.tiff'
    is_annotation = false
    ...


``path`` leads to the file to be loaded. ``type`` defines which file type is to be loaded. If ``type`` is not supplied 
GreedyFHist guesses the filetype based on the file ending: ``csv`` are loaded as ``pointsets`` and ``geojson`` as 
geojson data. Otherwise paths are treated as image data and loaded as images.

``is_annotation`` is set to ``false`` by default. If set to ``true``, ``Nearest Neighbor`` interpolation is used instead of 
``Linear`` interpolation. Also multichannel annotation images are treated as C x W x H instead of W x H x C. 


Pointset data
=============

Below see a full example for configuring pointset data.

.. code-block::

    ...
    [input.additional_data]

    path = 'pointset.csv'
    x_axis = 'x'
    y_axis = 'y'
    index_col = None
    header = None
    ...


Pointsets are internally parsed as pandas DataFrames. ``x_axis`` is the column used to index x-coordinates. ``y_axis`` indexes 
y-coordinates. ``index_col`` denotes the column used as the row index and ``header`` denotes the row used as the 
header. ``index_col`` and ``header`` are passed directly to pandas's ``pd.read_csv`` function.


Geojson data
============

Full example.

.. code-block::

    [input.additional_data]
    path = 'annotation.geojson'


Geojson data is defined using ``path``.


Spatial objects can be combined to composite spatial objects (see pairwise and groupwise examples).