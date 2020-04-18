"""
This file contain code, that help visualize importance weight distribution
"""

import plotly.graph_objs as go
from plotly.offline import plot, iplot
import cv2
import base64

def plot_distribution(img_path):
    """
    Function draw class distribution  for image
    Parameters:
    -----------
    img_path : str
        Path to input picture.

    """
    img = Image.open(img_path)
    img_width, img_height = img.size
    img = prepare_image(img = img)
    model = vgg19(pretrained=True).cuda().eval() 
    predict = model.forward(img)
    predict = predict.detach().cpu().numpy().reshape(-1)
    
    label = pd.read_csv('./label.csv', sep = ';', index_col=0)
    label['predict'] = predict
    label.sort_values(by = 'predict', inplace = True)
    trace = go.Bar(x = [str(i) + '_' + j for i, j in enumerate(label.label)], y = label.predict)
    l = go.Layout(
        title = 'Class distribution',
        xaxis = dict(
            title = 'Class'
        ),
        yaxis = dict(
            title = 'Score'
        )
    )
    fig = go.Figure(data = [trace], layout = l)
    iplot(fig)
    
def visual_importance(scale_factor, patch_size, img_path, name, mean, save_as_pic = False):
    """
    Function draw importance distribution on image.
    Parameters:
    -----------
    scale_factor : int
        How match compress image from zero to one.
    patch_size : int 
        Size of patch, should be dividers of 28.
    img_path : str
        Path to input picture.
    name : str
        Name of result picture, that would be saved in report directory.
    mean : bool
        How we should erase pixels? If False then erase by zero, else by mean.
    save_as_pic : bool
        Should we save image as picture or plot as figure.
        If True, than save as picture in report directory.
    """
    img = Image.open(img_path)
    img_width, img_height = img.size
    img = prepare_image(img)
    
    model = eval_importance().cuda()
    res = model.forward(img, 0, 0, 28//patch_size, mean)
    importance = np.array([float(torch.norm(res[0] - res[k + 1])) for k in range((28//patch_size)**2)])    

    shapes = []
    for i in range(28 // patch_size):
        for j in range(28 // patch_size):
            k = (28 // patch_size) * i + j
            shapes += [
                {
                    'type': 'rect',
                    'x0': j * img_width*scale_factor / (28 // patch_size),
                    'y0': ((28 // patch_size) - 1 - i) * img_height * scale_factor / (28 // patch_size),
                    'x1': (j + 1) * img_width*scale_factor / (28 // patch_size),
                    'y1': ((28 // patch_size) - i) * img_height * scale_factor / (28 // patch_size),
                    'line': {
                        'color': 'white' ,
                        'width': 2,
                    },
                    'fillcolor': 'rgba(255, 255, 255,' + str(0.8 * importance[k] / importance.max()) + ')'
                }
            ]
    
    with open(img_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    #add the prefix that plotly will want when using the string as source
    encoded_image = "data:image/png;base64," + encoded_string

    layout = go.Layout(
        shapes = shapes,
        xaxis = go.layout.XAxis(
            visible = False,
            range = [0, img_width*scale_factor]),
        yaxis = go.layout.YAxis(
            visible=False,
            range = [0, img_height*scale_factor],
            # the scaleanchor attribute ensures that the aspect ratio stays constant
            scaleanchor = 'x'),
        width = img_width*scale_factor,
        height = img_height*scale_factor,
        margin = {'l': 0, 'r': 0, 't': 0, 'b': 0},
        images = [dict(
            x=0,
            sizex=img_width*scale_factor,
            y=img_height*scale_factor,
            sizey=img_height*scale_factor,
            xref="x",
            yref="y",
            opacity=1.0,
            layer="below",
            sizing="stretch",
            source=encoded_image
        )]
    )

    # we add a scatter trace with data points in opposite corners to give the Autoscale feature a reference point
    fig = go.Figure(data=[{
        'x': [0, img_width * scale_factor], 
        'y': [0, img_height * scale_factor], 
        'mode': 'markers',
        'marker': {'opacity': 0}}
        ],layout = layout
        )
    if save_as_pic:
        fig.write_image(f'./report/pictures/{name}_{patch_size}.png')
    else:
        iplot(fig)

def compare_batch_size(img_path, name, mean = False, patch_sizes = [14, 7, 4, 2, 1]):
    """
    Function allows compare different patch sizes in terms of important
    
    Parameters:
    ----------
    img_path : str
        Path to input picture
    name : str
        Name of result picture, that would be saved in report directory
    mean : bool
        How we should erase pixels? If False then erase by zero, else by mean. 
    patch_sizes : list
        List of patch sizes. Every patch size should be dividers of 28.
    """
    for patch_size in patch_sizes:
         visual_importance(
             scale_factor = 1,
             patch_size = patch_size,
             img_path = img_path,
             name = name,
             mean = mean,
             save_as_pic = True
         )
    
    img = cv2.imread(img_path)
    
    for patch_size in patch_sizes:
        img = np.concatenate(
            (img, cv2.imread(f"./report/pictures/{name}_{patch_size}.png")),
            axis = 1
        )
    img = Image.fromarray(img[:, :, ::-1], 'RGB')
    img.save(f'./report/{name}.png')
