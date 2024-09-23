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
@click.option('--config', '-c', type=click.Path())
def groupwise_registration(config):
    cmdln_processor.groupwise_registration(config)


@click.command()
@click.option('--moving-image', '-m', type=click.Path())
@click.option('--fixed-image', '-f', type=click.Path())
@click.option('--output-directory', '-o', type=click.Path())
@click.option('--moving-mask', '-mmask', type=click.Path())  # Make this optional
@click.option('--fixed-mask', '-fmask', type=click.Path())  # Make this optional
@click.option('--path-to-greedy', '-g', type=click.Path())
@click.option('--config', '-c', type=click.Path())
@click.option('--images', type=click.Path(), multiple=True)
@click.option('--annotations', type=click.Path(), multiple=True)
@click.option('--pointsets', type=click.Path(), multiple=True)
@click.option('--geojsons', type=click.Path(), multiple=True)
def register(moving_image=None,
             fixed_image=None,
             output_directory=None,
             moving_mask=None,
             fixed_mask=None,
             path_to_greedy=None,
             config=None,
             images=None,
             annotations=None,
             pointsets=None,
             geojsons=None):
    cmdln_processor.register(
        moving_image_path=moving_image,
        fixed_image_path=fixed_image,
        output_directory=output_directory,
        moving_mask_path=moving_mask,
        fixed_mask_path=fixed_mask,
        path_to_greedy=path_to_greedy,
        config_path=config,
        images=images,
        annotations=annotations,
        pointsets=pointsets,
        geojsons=geojsons
    )

# TODO: Fix this. Should only be needed for 
@click.command()
@click.option('--transformation', '-t', type=click.Path(), required=True)
@click.option('--output-directory', '-o', type=click.Path(), default='out')
@click.option('--config', '-c', type=click.Path(), required=False)
@click.option('--images', type=click.Path(), multiple=True)
@click.option('--annotations', type=click.Path(), multiple=True)
@click.option('--pointsets', type=click.Path(), multiple=True)
@click.option('--geojsons', type=click.Path(), multiple=True)
def transform(transformation=None,
         output_directory=None,
         config=None,
         images=None,
         annotations=None,
         pointsets=None,
         geojsons=None):
    cmdln_processor.apply_transformation(output_directory=output_directory,
                                         config=config,
                                         images=images,
                                         annotations=annotations,
                                         pointsets=pointsets,
                                         geojsons=geojsons,                                         
                                         registerer=None,
                                         registration_result=None,
                                         registration_result_path=transformation)



cli.add_command(register)
cli.add_command(transform)
cli.add_command(groupwise_registration)


if __name__ == '__main__':
    cli()