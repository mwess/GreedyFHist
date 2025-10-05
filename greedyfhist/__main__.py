import click

import greedyfhist.cmdln_processor as cmdln_processor
from greedyfhist.cmdln_processor import (
    ImageConfig,
    PointsetConfig
)


class SpatialSectionConfigParamType(click.ParamType):
    name = "SpatialSection"
    def convert(self, value, param, ctx):
        pass


class ImageConfigParamType(click.ParamType):
    name = "Image"
    def convert(self, value, param, ctx):
        return ImageConfig.parse_from_cmd_str(value, requires_image=True)


class ImageMaskConfigParamType(click.ParamType):
    name = "ImageMask"
    
    def convert(self, value, param, ctx):
        return ImageConfig.parse_from_cmd_str(value, requires_image=False)

        
class PointsetConfigParamType(click.ParamType):
    name = "Pointset"
    def convert(self, value, param, ctx):
        return PointsetConfig.parse_from_cmd_str(value)


@click.group()
@click.version_option()
def cli():
    pass


@cli.command('register')
@click.option('-m',
              '-mov', 
              '--moving-image',
              '--moving',
              type=ImageConfigParamType(), 
              help='Moving image params.')
@click.option('-f',
              '-fix',
              '--fixed-image',
              '--fixed',
              type=ImageConfigParamType(), 
              help='Fixed image params')
@click.option('-o',
              '-out',
              '--output',
              '--output-directory',
              type=click.Path(),
              required=False,
              default='out',
              help='Output directory.')
@click.option('-mm',
              '-mmask',
              '--moving-mask',
              '--moving-image-mask',
              type=ImageMaskConfigParamType(),
              required=False,
              help='Path to optional mask for moving image.')
@click.option('-fm',
              '-fmask',
              '--fixed-mask',
              '--fixed-image-mask',
              type=ImageMaskConfigParamType(),
              required=False,
              help='Path to optional mask for fixed image.')
@click.option('-g',
              '--path-to-greedy',
              '--greedy',
              type=click.Path(),
              required=False,
              default='',
              help='Path to Greedy. If not provided assumes that Greedy is in the PATH environment variable.')
@click.option('-d',
              '--use-docker-executable',
              type=click.BOOL,
              default=False,
              help='If the greedy is used as a docker executable, set this to True.')
@click.option('-i',
              '--images',
              type=ImageConfigParamType(),
              required=False,
              multiple=True,
              help='Additional images to apply the transformation to.')
@click.option('-p',
              '--pointsets',
              type=PointsetConfigParamType(),
              required=False,
              multiple=True,
              help='Additional pointsets to apply the transformation to.')
@click.option('-gs',
              '--geojsons',
              type=click.Path(),
              required=False,
              multiple=True,
              help='Additional geojsons to apply the transformation to.')
@click.option('-c',
              '--config',
              '--config_path',
              type=click.Path(),
              required=False,
              help='Additional config parameters. Any parameters placed in config will override commandline arguments.')
def register(moving_image,
             fixed_image,
             output,
             moving_mask,
             fixed_mask,
             path_to_greedy,
             use_docker_executable,
             images,
             pointsets,
             geojsons,
             config):
    cmdln_processor.register(
        moving_image_config=moving_image,
        fixed_image_config=fixed_image,
        output_directory=output,
        moving_mask_config=moving_mask,
        fixed_mask_config=fixed_mask,
        path_to_greedy=path_to_greedy,
        use_docker_executable=use_docker_executable,
        images=images,
        pointsets=pointsets,
        geojsons=geojsons,
        config_path=config,
    )
    #click.echo(f'Moving image: {moving_image}')
    #click.echo(f'Fixed image: {fixed_image}')




# TODO: Explicit arguments for groupwise registration is missing
# IDEA: Write image with associated annotations in a list.
# Then just register all of that and write in separate directories as a result.
@click.command()
@click.option('-m',
              '-mov', 
              '--moving-image',
              '--moving',
              type=ImageConfigParamType(), 
              multiple=True,
              help='Moving image params.')
@click.option('-f',
              '-fix',
              '--fixed-image',
              '--fixed',
              type=ImageConfigParamType(), 
              help='Fixed image params')
@click.option('-o',
              '-out',
              '--output',
              '--output-directory',
              type=click.Path(),
              required=False,
              default='out',
              help='Output directory.')
@click.option('-mm',
              '-mmask',
              '--moving-mask',
              '--moving-image-mask',
              type=ImageMaskConfigParamType(),
              multiple=True,
              required=False,
              help='Path to optional mask for moving image. Will be applied in the same order as moving images.')
@click.option('-fm',
              '-fmask',
              '--fixed-mask',
              '--fixed-image-mask',
              type=ImageMaskConfigParamType(),
              required=False,
              help='Path to optional mask for fixed image.')
@click.option('-g',
              '--path-to-greedy',
              '--greedy',
              type=click.Path(),
              required=False,
              help='Path to Greedy. If not provided assumes that Greedy is in the PATH environment variable.')
@click.option('-d',
              '--use-docker-executable',
              type=click.BOOL,
              default=False,
              help='If the greedy is used as a docker executable, set this to True.')
@click.option('-c',
              '--config',
              '--config_path',
              type=click.Path(),
              required=False,
              help='Additional config parameters. Any parameters placed in config will override commandline arguments.')
def groupwise_registration(
             moving_image,
             fixed_image,
             output,
             moving_mask,
             fixed_mask,
             path_to_greedy,
             use_docker_executable,
             config):
    cmdln_processor.groupwise_registration(
        moving_images_config=moving_image,
        fixed_image_config=fixed_image,
        output_directory=output,
        moving_masks_config=moving_mask,
        fixed_mask_config=fixed_mask,
        path_to_greedy=path_to_greedy,
        use_docker_executable=use_docker_executable,
        config=config)


# TODO: Fix this. Should only be needed for 
@click.command()
@click.option('-o',
              '-out',
              '--output',
              '--output-directory',
              type=click.Path(),
              required=False,
              default='out',
              help='Output directory.')
@click.option('-t',
              '--transformation', 
              type=click.Path(), 
              required=True)
@click.option('-i',
              '--images',
              type=ImageConfigParamType(),
              required=False,
              multiple=True,
              help='Additional images to apply the transformation to.')
@click.option('-p',
              '--pointsets',
              type=PointsetConfigParamType(),
              required=False,
              multiple=True,
              help='Additional pointsets to apply the transformation to.')
@click.option('-gs',
              '--geojsons',
              type=click.Path(),
              required=False,
              multiple=True,
              help='Additional geojsons to apply the transformation to.')
@click.option('-c',
              '--config',
              '--config_path',
              type=click.Path(),
              required=False,
              help='Additional config parameters. Any parameters placed in config will override commandline arguments.')
@click.option('-c',
              '--config', 
              type=click.Path(), 
              required=False)
def transform(output,
              transformation,
              images,
              pointsets,
              geojsons,
              config):
    cmdln_processor.apply_transformation(output_directory=output,
                                         path_to_transform=transformation,
                                         images=images,
                                         pointsets=pointsets,
                                         geojsons=geojsons,
                                         config_path=config)


cli.add_command(register)
cli.add_command(transform)
cli.add_command(groupwise_registration)


if __name__ == '__main__':
    cli()