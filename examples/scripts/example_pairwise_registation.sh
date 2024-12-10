# # Start with registration.
echo "Starting with registration..."
greedyfhist register \
    --moving-image ../pairwise_examples/images/moving_image.ome.tif \
    --fixed-image ../pairwise_examples/images/fixed_image.ome.tif \
    --config example_registration.toml \
    --pointsets ../pairwise_examples/annotations/moving_pointset.csv \


# Now do warping. Its technically redundant, but we do it to show
# the functionality of transform.
echo "Transforming data..."
greedyfhist transform --transformation out/transformation/registration \
    --output-directory pairwise-transform-example \
    --images ../pairwise_examples/images/moving_image.ome.tif \
    --pointsets ../pairwise_examples/annotations/moving_pointset.csv