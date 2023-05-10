"""
    File for testing whether or not we can recover
    the images from the dataset. 
"""
from matplotlib import pyplot, image
import brainscore

neural_data = brainscore.get_assembly(name="dicarlo.MajajHong2015.public")
print(neural_data)

stimulus_set = neural_data.attrs['stimulus_set']

stimulus_path = stimulus_set.get_stimulus(stimulus_set['stimulus_id'][0])

img = image.imread(stimulus_path)

pyplot.imshow(img)
pyplot.savefig("plots/test_load_image.png")