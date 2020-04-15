# Style transfer with preservation of the central objects of the scene.
Add styles from famous paintings to any photo with preservation of the central objects of the scene.

<p align = 'center'>
<img src = 'Images/githubpic.jpg' height = '700px'>
</p>

## Documentation

### Stylizing photo
Use `style.py` to transfer style into a photo. Run `python style.py -h` to view all the possible parameters or you can see them below
- `img_path` : Path to content image.
- `style_img_path` : Path to style image.
- `directory` : Directory where result image would be saved.
- `name` : Name of result image.
- `weight_function` : Type of function, that would be used for finding weight matrix.
           Possible values: _patch_ -- simple patch method;
            _moving_patch_ -- moving patch method;
            _superpixel_ -- superpixel method;
            _segmentation_ -- segmentation method;
            _gatys_ -- standard Gatys algorithm.

-  `contrast` : Contrast coefficient, optional parameter. Default value equal to one.
      Bigger value means, that front object will have bigger values in weight matrix.
-  `img_size` : Minimal size side of resulting picture, optional parameter.
-  `grid_size` : Size of grid grid_size x grid_size, optional parameter.

Example usage:

    python style.py Images/content.jpg Images/style.jpg /report/ Stylization segmentation





### Requirements
You will need the following to run the above:
 - [pytorch](https://github.com/pytorch/pytorch)
 - [segmentation_models_pytorch](https://github.com/qubvel/segmentation_models.pytorch)
 - [plotly](https://github.com/plotly/plotly.py)
 - [albumentations](https://github.com/albumentations-team/albumentations)
 - [catalyst](https://github.com/catalyst-team/catalyst)
 - Nvidia GPU. All results were produced on the video card NVIDIA RTX 2080 super with 8GB memory.
 
 
