# Style transfer with preservation of the central objects of the scene.
Add styles from famous paintings to any photo with preservation of the central objects of the scene.

<p align = 'center'>
<img src = 'Images/githubpic.jpg' height = '700px'>
</p>

## Documentation

### Stylizing photo
Use `style.py` to transfer style into a photo (don't forget to dowload pretrained models). Run `python style.py -h` to view all the possible parameters or you can see them below
- `img_path` : Path to content image.
- `style_img_path` : Path to style image.
- `directory` : Directory where result image would be saved.
- `name` : Name of result image.
- `weight_function` : Type of function, that would be used for finding weight matrix.
           Possible values: _patch_ &mdash; simple patch method;
            _moving_patch_ &mdash; moving patch method;
            _superpixel_ &mdash; superpixel method;
            _segmentation_ &mdash; segmentation method;
            _gatys_ &mdash; standard Gatys algorithm.

-  `contrast` : Contrast coefficient, optional parameter. Default value equal to one.
      Bigger value means, that front object will have bigger values in weight matrix.
-  `img_size` : Minimal size side of resulting picture, optional parameter. Bigger size require a lot of GPU memory.
-  `grid_size` : Size of grid grid_size x grid_size for patch and moving patch methods, optional parameter.

Example usage:

    python style.py .\Images\content.jpg .\Images\style.jpg .\report\ Stylization segmentation

    

### Requirements
You will need the following to run code:
 - Dowload pretrained [segmentation model](https://yadi.sk/d/hcNhRltixuxIVw) and [vgg19 model](https://yadi.sk/d/yxJrNraRcujGCA) and put them to models directory.
 - [pytorch](https://github.com/pytorch/pytorch)
 - [segmentation_models_pytorch](https://github.com/qubvel/segmentation_models.pytorch)
 - [plotly](https://github.com/plotly/plotly.py)
 - [albumentations](https://github.com/albumentations-team/albumentations)
 - [catalyst](https://github.com/catalyst-team/catalyst)
 - Nvidia GPU. All results were produced on the video card NVIDIA RTX 2080 super with 8GB memory.
 
## Examples
<p align = 'center'>
<img src = 'Images/github_pic_2.jpg'>
</p>

You can see at the picture above stylization algorithm with different weight matrix. All this images were made with contrast coefficient 10. That should help to see difference between selected region. Segmentation method gives the best results on every picture. 

## Segmentation
In segmentation folder you can find notebook with training segmentation model on [this](https://github.com/victorkitov/BF_Style/tree/master/dataset) dataset.
