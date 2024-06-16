import click
import greedyfhist.cmdln_processor as cmdln_processor


@click.group()
@click.version_option()
def cli():
    pass


@click.command()
@click.option('--moving-image', type=click.Path(), required=True)
@click.option('--fixed-image', type=click.Path(), required=True)
@click.option('--output-directory', type=click.Path(), required=True, default='out')
@click.option('--moving-mask', type=click.Path())  # Make this optional
@click.option('--fixed-mask', type=click.Path())  # Make this optional
@click.option('--path-to-greedy', type=click.Path())
@click.option('--config', type=click.Path())
def register(*args, **kwargs):
    pass

# @click.command()
# @click.option('--moving-image', type=click.Path(), required=True)
# @click.option('--fixed-image', type=click.Path(), required=True)
# @click.option('--output-directory', type=click.Path(), required=True, default='out')
# @click.option('--moving-mask', type=click.Path())  # Make this optional
# @click.option('--fixed-mask', type=click.Path())  # Make this optional
# @click.option('--path-to-greedy', type=click.Path())
# @click.option()
# def register(moving_image,
#              fixed_image,
#              output_directory,
#              moving_mask=None,
#              fixed_mask=None,
#              path_to_greedy=None,
#              is_cmd_line=True):
#     cmdln_processor.register(
#         moving_image,
#         fixed_image,
#         output_directory,
#         moving_mask,
#         fixed_mask,
#         path_to_greedy,
#         is_cmd_line
#     )



@click.command()
@click.option('--transformation', type=click.Path(), required=True)
@click.option('--output-directory', type=click.Path(), default='warp_out')
@click.option('--path-to-greedy', type=click.Path(), default='')
@click.option('--images', type=click.Path(), multiple=True, default=[])
@click.option('--annotations', type=click.Path(), multiple=True, default=[])
@click.option('--coordinates', type=click.Path(), multiple=True, default=[])
@click.option('--geojsons', type=click.Path(), multiple=True, default=[])
def transform(transformation,
         output_directory,
         path_to_greedy,
         images,
         annotations,
         coordinates,
         geojsons):
    cmdln_processor.transform(transformation=transformation,
                              output_directory=output_directory,
                              images=images,
                              annotations=annotations,
                              coordinates=coordinates,
                              geojsons=geojsons)


@click.command()
@click.option('--config', type=click.Path(), required=True)
def register_by_config(config_path):
    cmdln_processor.register_by_config(config_path)


# TODO: Implement this.
@click.command()
def register_and_warp():
    pass


cli.add_command(register)
cli.add_command(transform)
cli.add_command(register_and_warp)
cli.add_command(register_by_config)


if __name__ == '__main__':
    cli()