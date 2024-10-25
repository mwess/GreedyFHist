"""
General utils files. Lowest level of utils. Cannot import from anywhere else in the project.
"""
import os
import shlex
import subprocess

from greedyfhist.options import AffineGreedyOptions, NonrigidGreedyOptions


def call_command(cmd: str):
    """
    Simple wrapper function around a command.
    :param cmd:
    :return:
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
    
    args = {}
    args['-rf'] = path_small_ref_img
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

    Returns:
        _type_: Return of command line execution.
    """
    if use_docker_container:
        abs_temp_directory = os.path.abspath(temp_directory)
        # v_option = f'$(pwd)/{abs_temp_directory}:/{temp_directory}'
        v_option = f'{abs_temp_directory}:/{temp_directory}'
        path_to_greedy = f'docker run -v {v_option} {path_to_greedy}'
    cost_fun_params = options.cost_function
    if options.cost_function == 'ncc' or options.cost_function == 'wncc':
        cost_fun_params += f' {options.kernel_size}x{options.kernel_size}'
    aff_rgs = {}
    aff_rgs['-d'] = '2'
    aff_rgs['-i'] = [path_to_fixed_image, path_to_moving_image]
    aff_rgs['-o'] = path_output
    aff_rgs['-m'] = cost_fun_params
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

    Returns:
        _type_: Return of command line execution.
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
    if options.use_sv:
        def_args['-sv'] = ''
        if options.exp is not None:
            def_args['-exp'] = options.exp
    elif options.use_svlb:
        def_args['-svlb'] = ''
        if options.exp is not None:
            def_args['-exp'] = options.exp
    if affine_pre_transform is None:
        def_args[ia[0]] = ia[1]

    def_cmd = build_cmd_string(path_to_greedy, def_args)
    def_ret = call_command(def_cmd)
    return def_ret