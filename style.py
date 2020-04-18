import argparse
import warnings
warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser(
    description='Style transfer with preservation of the central objects of the scene'
)

parser.add_argument("img_path", help="Path to content image", type = str)
parser.add_argument("style_img_path", help="Path to style image", type = str)
parser.add_argument("directory", help="Directory where result image would be saved", type = str)
parser.add_argument("name", help="Name of result image", type = str)
parser.add_argument("weight_function", type = str, help =\
    """
    Type of function, that would be used for finding weight matrix.
           Possible values: 'patch' : simple patch method;
            'moving_patch' : moving patch method;
            'superpixel' : superpixel method;
            'segmentation' : segmentation method;
            'gatys' : standard Gatys algorithm
    """
)

parser.add_argument(
    "--contrast",
    help = "Contrast coefficient. Default value equal to one.\
    Bigger value means, that front object will have bigger values in weight matrix.", type = int
)
parser.add_argument(
    "--img_size",
    help="Minimal size side of resulting picture", type = int
)
parser.add_argument(
    "-g", "--grid_size",
    help="Size of grid: grid_size x grid_size.\
    This parameters used in patch and moving_patch method", type = int
 )

args = parser.parse_args()
                    
from functions import *
    
stylization(
    img_path = args.img_path,
    style_img_path = args.style_img_path,
    directory = args.directory,
    name = args.name,
    weight_function = args.weight_function,
    contrast = args.contrast if args.contrast else 1,
    img_size = args.img_size if args.img_size else 640,
    grid_size = args.grid_size if args.grid_size else 4
)

print('Done')
