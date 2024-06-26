# Usage

GreedyFHist has two main registration options: `Pairwise registration` and `Groupwise registration`

## Pairwise registration

Pairwise registration can be performed in 3 ways: 2 command line options and interactively using GreedyFHist's API.

1. Commandline option 1:

Moving and fixed image are parsed as commandline arguments. GreedyFHist tries to guess the correct file type for moving and fixed image based on the file ending. Options for configuring the registration can be passed in a config file. Additional data types can be passed as well as images, annotations, pointsets and geojson files. After registration additionally provided data is transformed from the moving to the fixed image space and stored in `output-directory`. 


```
greedyfhist register \
    --moving-image moving_image.ome.tif \
    --fixed-image fixed_image.ome.tif \
    --config config.toml \
    --output-directory out
```

2. Commandline option 2:

An alternative way of using GreedyFHists commandline interface is by defining everying in a configuration `config.toml` file. If moving and fixed image are used on the commandline, contents of the `config.toml` are ignored.

```
greedyfhist register --config config.toml
```

3. Using GreedyFHist's API:




## Groupwise Registration

This option is used for registration of multiple stained images at once. Groupwise registration can be executed in two ways: Commandline option and by using the API. 

1. Commandline option:

```
greedyfhist groupwise-registration --config config.toml
```

2. API:

