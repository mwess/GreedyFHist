=============================
Pairwise registration example
============================= 

In this section we show examples of pairwise registration using the command line interface. We give one 
example using arguments and one example using a 
configuration file.

---------------------------
Using commandline arguments
---------------------------


Using the commandline option requires a moving and fixed image. Registration options can be 
configured in the ``example_registration.toml`` file. Additional 
data for transformations can be passed as ``imagew``, ``annotationw``, ``pointsets``, ``geojsons``. 
GreedyFHist supports most common image formats, but if any are missing, feel free to post 
an issue. `annotations` are annotation masks with the last image channel used to denote classes, e.g., W x H x C, 
if a multichannel image is used.

An example call could look like this:

.. code-block:: bash

   greedyfhist register \
      --moving-image ../pairwise_examples/images/moving_image.ome.tif \
      --fixed-image ../pairwise_examples/images/fixed_image.ome.tif \
      --config example_registration.toml \
      --annotations ../pairwise_examples/annotations/moving_annotation.ome.tiff \
      --pointsets ../pairwise_examples/annotations/moving_pointset.csv \
      --geojsons ../pairwise_examples/annotations/moving_annotation.geojson                            


-----------------
Using config.toml
-----------------

Another option for using the commandline is using the ``configuration.toml`` without any additional 
arguments. If moving and fixed image are supplied as command line arguments, any input data 
in ``[input]`` in the ``configuration.toml`` is ignored. Sections ``gfh_options`` and ``options`` are 
explained in config_example. The ``input`` section can be formulated as follows:


.. code-block::

    ...
    [input]

    [input.moving_image]
    [input.moving_image.reference_image]

    path = '../pairwise_examples/images/moving_image.ome.tif'

    [[input.moving_image.additional_data]]

    path = '../pairwise_examples/annotations/some_annotation.ome.tif'
    is_annotation = true

    [[input.moving_image.additional_data]]

    path = '../pairwise_examples/annotation/more_annotation.geojson'

    [input.fixed_image.reference_image]

    path = '../pairwise_examples/images/fixed_image.ome.tif'
    ...


``input.moving_image.reference_image`` and ``input.fixed_image.reference_image`` sections need to be defined. Additional data for 
transforming can be supplied as ``[[input.moving_image.additional_data]]``. 
Examples for defining sections and additional data types in the configuration file can be found here in the `config` section.

A full example can be found further below. 

Using this example we can run ``greedyfhist register -c configuration.toml`` and get the following output:


.. code-block::

    out/
    ├── registrations
    │   ├── registration_transform
    │   │   ├── fixed_transform
    │   │   │   ├── attributes.json
    │   │   │   └── transform.txt
    │   │   └── moving_transform
    │   │       ├── attributes.json
    │   │       └── transform.txt
    │   └── reverse_registration_transform
    │       ├── fixed_transform
    │       │   ├── attributes.json
    │       │   └── transform.txt
    │       └── moving_transform
    │           ├── attributes.json
    │           └── transform.txt
    └── transformed_data
        ├── moving_image.ome.tif
        ├── moving_pointset.csv
        └── preprocessing_data
            ├── fixed_mask.png
            └── moving_mask.png        


``registrations`` contains the transformation from moving to fixed_image space. ``transformed_data`` contains the transformed moving image and 
additionally transformed data. If possible, masks used during preprocessing are also stored. GreedyFHist stores transformed image data as 
`ome.tif` files using `pyvips`.

Full example configuration.

.. code-block::

    [options]

    output_directory = 'out'
    path_to_greedy = ''


    [gfh_options]

    pre_sampling_factor = 0.25
    pre_sampling_auto_factor = 3500
    do_affine_registration = true
    do_nonrigid_registration = true
    temporary_directory = 'tmp'
    remove_temporary_directory = true
    yolo_segmentation_min_size = 5000

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
    resolution = '1024x1024'
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


    [input]

    [input.moving_image]
    [input.moving_image.reference_image]

    path = '../pairwise_examples/images/moving_image.ome.tif'
    type = 'tif'

    [[input.moving_image.additional_data]]

    path = '../pairwise_examples/annotations/some_annotation.ome.tif'
    type = 'tif'
    is_annotation = true
    keep_axis = false

    [[input.moving_image.additional_data]]

    path = '../pairwise_examples/annotation/more_annotation.geojson'


    [input.fixed_image.reference_image]

    path = '../pairwise_examples/images/fixed_image.ome.tif'
    type = 'tif'


--------------------------------
Using interactive Python session
--------------------------------

An example using the interactive Python session can be found in `examples/notebooks/pairwise.ipynb <https://github.com/mwess/GreedyFHist/blob/master/examples/notebooks/pairwise.ipynb>`_.