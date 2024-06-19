import click
import greedyfhist.cmdln_processor as cmdln_processor


@click.group()
@click.version_option()
def cli():
    pass


# TODO: Groupwise registration is missing
# IDEA: Write image with associated annotations in a list.
# Then just register all of that and write in separate directories as a result.
@click.command()
@click.option('--source_directory', type=click.Path())
@click.option('--output_directory', type=click.Path())
@click.option('--config', type=click.Path())
def groupwise_registration(source_directory,
                           output_directory,
                           config):
    cmdln_processor.groupwise_registration(source_directory, output_directory, config)



@click.command()
@click.option('--moving-image', type=click.Path())
@click.option('--fixed-image', type=click.Path())
@click.option('--output-directory', type=click.Path(), default='out')
@click.option('--moving-mask', type=click.Path())  # Make this optional
@click.option('--fixed-mask', type=click.Path())  # Make this optional
@click.option('--path-to-greedy', type=click.Path())
@click.option('--config', type=click.Path())
@click.option('--transform-images', type=click.Path(), multiple=True, default=[])
@click.option('--transform-annotations', type=click.Path(), multiple=True, default=[])
@click.option('--transform-pointsets', type=click.Path(), multiple=True, default=[])
@click.option('--transform-geojsons', type=click.Path(), multiple=True, default=[])
def register(moving_image,
             fixed_image,
             output_directory,
             moving_mask=None,
             fixed_mask=None,
             path_to_greedy=None,
             config=None,
             images=None,
             annotations=None,
             pointsets=None,
             geojsons=None):
    cmdln_processor.register(
        moving_image,
        fixed_image,
        output_directory,
        moving_mask,
        fixed_mask,
        path_to_greedy,
        config,
        images,
        annotations,
        pointsets,
        geojsons
    )

# TODO: Fix this. Should only be needed for 
@click.command()
@click.option('--transformation', type=click.Path(), required=True)
@click.option('--output-directory', type=click.Path(), default='warp_out')
@click.option('--images', type=click.Path(), multiple=True, default=[])
@click.option('--annotations', type=click.Path(), multiple=True, default=[])
@click.option('--coordinates', type=click.Path(), multiple=True, default=[])
@click.option('--geojsons', type=click.Path(), multiple=True, default=[])
@click.option('--config', type=click.Path(), required=False)
def transform(transformation,
         output_directory,
         images,
         annotations,
         coordinates,
         geojsons,
         config):
    cmdln_processor.apply_transformation(output_directory=output_directory,
                                         images=images,
                                         annotations=annotations,
                                         coordinates=coordinates,
                                         geojsons=geojsons,
                                         config=config,
                                         registerer=None,
                                         registration_result=None,
                                         registration_result_path=transformation)



cli.add_command(register)
cli.add_command(transform)


if __name__ == '__main__':
    cli()