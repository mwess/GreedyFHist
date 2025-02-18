"""
General utils files. Lowest level of utils. Cannot import from anywhere else in the project.
"""
import os
import shlex
import subprocess

import numpy, numpy as np
import SimpleITK, SimpleITK as sitk

from greedyfhist.options import AffineGreedyOptions, NonrigidGreedyOptions


def call_command(cmd: str) -> subprocess.CompletedProcess:
    """
    Call a command and return its output.

    Args:
        cmd (str): Command to execute.

    Returns: CompletedProcess object.
    """
    ret = subprocess.run(shlex.split(cmd), capture_output=True)
    return ret


def build_cmd_string(path_to_exec: str, args: dict[str, str | list[str]]) -> str:
    """Small custom function for collection arguments in a function call."""
    cmd = [path_to_exec]
    for key in args:
        cmd.append(key)
        val = args[key]
        if val != '':
            if isinstance(val, list):
                cmd += val
            else:
                cmd.append(val)
    cmd = ' '.join([str(x) for x in cmd])
    return cmd


def composite_warps(path_to_greedy: str,
                    path_small_affine: str | None,
                    path_small_warp: str | None,
                    path_small_ref_img: str,
                    path_small_composite_warp: str,
                    invert=False) -> subprocess.CompletedProcess:
    """Calls greedy to composite transformations.

    Args:
        path_to_greedy (str):
        path_small_affine (str | None): Path to affine transform.
        path_small_warp (str | None): Path to displacement field.
        path_small_ref_img (str): Path to reference image.
        path_small_composite_warp (str): Path to output composite displacement field.
        invert (bool, optional): If True, inverts to order of composition. Defaults to False.

    Returns:
        subprocess.CompletedProcess: 
    """
    
    args = {'-rf': path_small_ref_img}
    if invert:
        transform_paths = []
        if path_small_affine is not None:
            transform_paths.append(f'{path_small_affine},-1')
        if path_small_warp:
            transform_paths.append(path_small_warp)
        args['-r'] = transform_paths
    else:
        transform_paths = []
        if path_small_warp is not None:
            transform_paths.append(path_small_warp)
        if path_small_affine is not None:
            transform_paths.append(path_small_affine)
        args['-r'] = transform_paths
    args['-rc'] = path_small_composite_warp
    cmd = build_cmd_string(path_to_greedy, args)
    ret = call_command(cmd)
    return ret


def affine_registration(path_to_greedy: str,
                        path_to_fixed_image: str,
                        path_to_moving_image: str,
                        path_output: str,
                        offset:int,
                        ia:str,
                        options: AffineGreedyOptions,
                        use_docker_container: bool = False,
                        temp_directory: str = ''
                        ) -> subprocess.CompletedProcess:
    """Calls greedy's affine registration function.

    Args:
        path_to_greedy (str): _description_
        path_to_fixed_image (str): _description_
        path_to_moving_image (str): _description_
        path_output (str): _description_
        offset (int): _description_
        ia (str): _description_
        options (GreedyOptions): _description_
        use_docker_container: bool: _description_. Defaults to False.
        temp_directory (str, optional): _description_. Defaults to ''.

    Returns:
        subprocess.CompletedProcess: Return of command line execution.
    """
    if use_docker_container:
        abs_temp_directory = os.path.abspath(temp_directory)
        v_option = f'{abs_temp_directory}:/{temp_directory}'
        path_to_greedy = f'docker run -v {v_option} {path_to_greedy}'
    cost_fun_params = options.cost_function
    if options.cost_function == 'ncc' or options.cost_function == 'wncc':
        cost_fun_params += f' {options.kernel_size}x{options.kernel_size}'
    aff_rgs = {'-d': '2',
               '-i': [path_to_fixed_image, path_to_moving_image],
               '-o': path_output,
               '-m': cost_fun_params}
    pyramid_iterations = 'x'.join([str(x) for x in options.iteration_pyramid])
    # aff_rgs['-n'] = f'{options.pyramid_iterations[0]}x{options.pyramid_iterations[1]}x{options.pyramid_iterations[2]}'
    aff_rgs['-n'] = pyramid_iterations
    aff_rgs['-threads'] = options.n_threads
    aff_rgs['-dof'] = str(options.dof)
    aff_rgs['-search'] = f'{options.rigid_iterations} 180 {offset}'.split()  # Replaced 360 with any for rotation parameter
    aff_rgs['-gm-trim'] = f'{options.kernel_size}x{options.kernel_size}'
    aff_rgs['-a'] = ''  # Doesnt get param how to parse?
    aff_rgs[ia[0]] = ia[1]

    aff_cmd = build_cmd_string(path_to_greedy, aff_rgs)
    aff_ret = call_command(aff_cmd)
    return aff_ret


def deformable_registration(path_to_greedy: str,
                            path_fixed_image: str,
                            path_moving_image: str,
                            options: NonrigidGreedyOptions,
                            output_warp: str | None = None,
                            output_inv_warp: str | None = None,
                            affine_pre_transform: str | None = None,
                            ia: tuple[str, str] = None,
                            use_docker_container: bool = False,
                            temp_directory: str = ''
                            ) -> subprocess.CompletedProcess:
    """Calls the deformable registration command of greedy.

    Args:
        path_to_greedy (str): 
        path_fixed_image (str): 
        path_moving_image (str): 
        options (GreedyOptions): Contains options to pass to greedy.
        output_warp (str, optional): Defaults to None.
        output_inv_warp (str, optional): Defaults to None.
        affine_pre_transform (_type_, optional): Contains path to affine_pre_transform. Necessary if ia is ia-com-init. Defaults to None.
        ia (tuple[str, str]):
        use_docker_container: bool. Defaults to False.
        temp_directory (str, optional): Defaults to ''.

    Returns:
        subprocess.CompletedProcess: Return of command line execution.
    """
    if use_docker_container:
        abs_temp_directory = os.path.abspath(temp_directory)
        v_option = f'{abs_temp_directory}:/{temp_directory}'
        path_to_greedy = f'docker run -v {v_option} {path_to_greedy}'
    cost_fun_params = options.cost_function
    if options.cost_function == 'ncc' or options.cost_function == 'wncc':
        cost_fun_params += f' {options.kernel_size}x{options.kernel_size}'
    def_args = {}
    if affine_pre_transform is not None:
        def_args['-it'] = affine_pre_transform
    def_args['-d'] = options.dim
    def_args['-m'] = cost_fun_params
    def_args['-i'] = [path_fixed_image, path_moving_image]
    pyramid_iterations = 'x'.join([str(x) for x in options.iteration_pyramid])
    def_args['-n'] = pyramid_iterations
    def_args['-threads'] = options.n_threads
    def_args['-s'] = [f'{options.s1}vox', f'{options.s2}vox']
    def_args['-o'] = output_warp
    def_args['-oinv'] = output_inv_warp
    if options.use_gm_trim:
        def_args['-gm-trim'] = f'{options.kernel_size}x{options.kernel_size}'
    if options.use_sv:
        def_args['-sv'] = ''
        if options.exp is not None:
            def_args['-exp'] = options.exp
    elif options.use_svlb:
        def_args['-svlb'] = ''
        if options.exp is not None:
            def_args['-exp'] = options.exp
    if options.tscale is not None:
        def_args['-tscale'] = options.tscale
    if affine_pre_transform is None:
        def_args[ia[0]] = ia[1]

    def_cmd = build_cmd_string(path_to_greedy, def_args)
    def_ret = call_command(def_cmd)
    return def_ret


def composite_sitk_transforms(transforms: list[SimpleITK.SimpleITK.Transform]) -> SimpleITK.SimpleITK.Transform:
    """Composites all Transforms into one composite transform.

    Args:
        transforms (List[SimpleITK.SimpleITK.Transform]): 

    Returns:
        SimpleITK.SimpleITK.Transform: 
    """
    composited_transform = sitk.CompositeTransform(2)
    for transform in transforms:
        composited_transform.AddTransform(transform)
    return composited_transform


def compose_reg_transforms(transform: SimpleITK.Transform,
                           moving_preprocessing_params: dict,
                           fixed_preprocessing_params: dict) -> SimpleITK.Transform:
    """Pre- and appends preprocessing steps from moving and fixed image as transforms to forward
    affine/nonrigid registration.

    Args:
        transform (SimpleITK.Transform): Computed affine/nonrigid registration
        moving_preprocessing_params (dict):
        fixed_preprocessing_params (dict):

    Returns:
        SimpleITK.SimpleITK.Transform: Composited end-to-end registration.
    """
    moving_padding = moving_preprocessing_params['padding']
    moving_cropping = moving_preprocessing_params['cropping_params']
    fixed_padding = fixed_preprocessing_params['padding']
    fixed_cropping = fixed_preprocessing_params['cropping_params']
    mov_ds_factor = moving_preprocessing_params['resampling_factor']
    fix_ds_factor = fixed_preprocessing_params['resampling_factor']
    
    all_transforms = sitk.CompositeTransform(2)

    pre_downscale_transform = sitk.ScaleTransform(2, (1/mov_ds_factor, 1/mov_ds_factor))
    post_upscale_transform = sitk.ScaleTransform(2, (fix_ds_factor, fix_ds_factor))
    
    aff_trans1 = sitk.TranslationTransform(2)
    offset_x = moving_cropping[2]
    offset_y = moving_cropping[0]
    aff_trans1.SetOffset((offset_x, offset_y))

    aff_trans2 = sitk.TranslationTransform(2)
    offset_x = -moving_padding[0]
    offset_y = -moving_padding[2]
    aff_trans2.SetOffset((offset_x, offset_y))
    
    aff_trans3 = sitk.TranslationTransform(2)
    aff_trans3.SetOffset((fixed_padding[0], fixed_padding[2]))
    
    aff_trans4 = sitk.TranslationTransform(2)
    aff_trans4.SetOffset((-fixed_cropping[2], -fixed_cropping[0]))

    all_transforms.AddTransform(pre_downscale_transform)
    all_transforms.AddTransform(aff_trans1)
    all_transforms.AddTransform(aff_trans2)
    all_transforms.AddTransform(transform)
    all_transforms.AddTransform(aff_trans3)
    all_transforms.AddTransform(aff_trans4)
    all_transforms.AddTransform(post_upscale_transform)
    return all_transforms


def compose_inv_reg_transforms(transform: SimpleITK.Transform,
                               moving_preprocessing_params: dict,
                               fixed_preprocessing_params: dict) -> SimpleITK.Transform:
    """Pre- and appends preprocessing steps from moving and fixed image as transforms to backward affine/nonrigid registration.  

    Args:
        transform (SimpleITK.Transform): Computed affine/nonrigid registration
        moving_preprocessing_params (dict):
        fixed_preprocessing_params (dict):

    Returns:
        SimpleITK.Transform: Composited end-to-end transform.
    """
    moving_padding = moving_preprocessing_params['padding']
    moving_cropping = moving_preprocessing_params['cropping_params']
    fixed_padding = fixed_preprocessing_params['padding']
    fixed_cropping = fixed_preprocessing_params['cropping_params']
    mov_ds_factor = moving_preprocessing_params['resampling_factor']
    fix_ds_factor = fixed_preprocessing_params['resampling_factor']
    
    all_transforms = sitk.CompositeTransform(2)

    pre_downscale_transform = sitk.ScaleTransform(2, (1/fix_ds_factor, 1/fix_ds_factor))
    post_upscale_transform = sitk.ScaleTransform(2, (mov_ds_factor, mov_ds_factor))

    aff_trans1 = sitk.TranslationTransform(2)
    offset_x = fixed_cropping[2]
    offset_y = fixed_cropping[0]
    aff_trans1.SetOffset((offset_x, offset_y))

    aff_trans2 = sitk.TranslationTransform(2)
    offset_x = -fixed_padding[0]
    offset_y = -fixed_padding[2]
    aff_trans2.SetOffset((offset_x, offset_y))
    

    aff_trans3 = sitk.TranslationTransform(2)
    aff_trans3.SetOffset((moving_padding[0], moving_padding[2]))
    
    aff_trans4 = sitk.TranslationTransform(2)
    aff_trans4.SetOffset((-moving_cropping[2], -moving_cropping[0]))

    all_transforms.AddTransform(pre_downscale_transform)
    all_transforms.AddTransform(aff_trans1)
    all_transforms.AddTransform(aff_trans2)
    all_transforms.AddTransform(transform)
    all_transforms.AddTransform(aff_trans3)
    all_transforms.AddTransform(aff_trans4)
    all_transforms.AddTransform(post_upscale_transform)
    return all_transforms


# TODO: There is probably a better way to ensure the dtype of the image.
def correct_img_dtype(img: numpy.ndarray) -> numpy.ndarray:
    """Changes the image type from float to np.uint8 if necessary.

    Args:
        img (numpy.ndarray): 

    Returns:
        numpy.ndarray: 
    """
    if np.issubdtype(img.dtype, np.floating):
        img = (img * 255).astype(np.uint8)
    else:
        img = img.astype(np.uint8)
    return img


def derive_sampling_factors(pre_sampling_factor: float | str,
                           max_size: int,
                           pre_sampling_max_img_size: int | None) -> tuple[float, float]:
    """Derives resampling factor for registration.

    Args:
        pre_sampling_factor (float | str): 
        max_size (int): 
        pre_sampling_max_img_size (int | None): 

    Returns:
        tuple[float, float]: 
    """
    if pre_sampling_factor == 'auto':
        if pre_sampling_max_img_size is not None:
            if max_size > pre_sampling_max_img_size:
                resampling_factor = pre_sampling_max_img_size / max_size
                moving_resampling_factor = resampling_factor
                fixed_resampling_factor = resampling_factor
            else:
                moving_resampling_factor = 1
                fixed_resampling_factor = 1
        else:
            moving_resampling_factor = 1
            fixed_resampling_factor = 1
    else:
        moving_resampling_factor = 1
        fixed_resampling_factor = 1
    return moving_resampling_factor, fixed_resampling_factor 