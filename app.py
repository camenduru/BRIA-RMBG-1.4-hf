import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import normalize
from huggingface_hub import hf_hub_download
import gradio as gr
from gradio_imageslider import ImageSlider
from briarmbg import BriaRMBG
import PIL
from PIL import Image
from typing import Tuple

net=BriaRMBG()
# model_path = "./model1.pth"
model_path = hf_hub_download("briaai/RMBG-1.4", 'model.pth')
if torch.cuda.is_available():
    net.load_state_dict(torch.load(model_path))
    net=net.cuda()
else:
    net.load_state_dict(torch.load(model_path,map_location="cpu"))
net.eval() 

    
def resize_image(image):
    image = image.convert('RGB')
    model_input_size = (1024, 1024)
    image = image.resize(model_input_size, Image.BILINEAR)
    return image


def process(image):

    # prepare input
    orig_image = Image.fromarray(image)
    w,h = orig_im_size = orig_image.size
    image = resize_image(orig_image)
    im_np = np.array(image)
    im_tensor = torch.tensor(im_np, dtype=torch.float32).permute(2,0,1)
    im_tensor = torch.unsqueeze(im_tensor,0)
    im_tensor = torch.divide(im_tensor,255.0)
    im_tensor = normalize(im_tensor,[0.5,0.5,0.5],[1.0,1.0,1.0])
    if torch.cuda.is_available():
        im_tensor=im_tensor.cuda()

    #inference
    result=net(im_tensor)
    # post process
    result = torch.squeeze(F.interpolate(result[0][0], size=(h,w), mode='bilinear') ,0)
    ma = torch.max(result)
    mi = torch.min(result)
    result = (result-mi)/(ma-mi)    
    # image to pil
    im_array = (result*255).cpu().data.numpy().astype(np.uint8)
    pil_im = Image.fromarray(np.squeeze(im_array))
    # paste the mask on the original image
    new_im = Image.new("RGBA", pil_im.size, (0,0,0,0))
    new_im.paste(orig_image, mask=pil_im)
    # new_orig_image = orig_image.convert('RGBA')

    return new_im
    # return [new_orig_image, new_im]


# block = gr.Blocks().queue()

# with block:
#     gr.Markdown("## BRIA RMBG 1.4")
#     gr.HTML('''
#       <p style="margin-bottom: 10px; font-size: 94%">
#         This is a demo for BRIA RMBG 1.4 that using
#         <a href="https://huggingface.co/briaai/RMBG-1.4" target="_blank">BRIA RMBG-1.4 image matting model</a> as backbone. 
#       </p>
#     ''')
#     with gr.Row():
#         with gr.Column():
#             input_image = gr.Image(sources=None, type="pil") # None for upload, ctrl+v and webcam
#             # input_image = gr.Image(sources=None, type="numpy") # None for upload, ctrl+v and webcam
#             run_button = gr.Button(value="Run")
            
#         with gr.Column():
#             result_gallery = gr.Gallery(label='Output', show_label=False, elem_id="gallery", columns=[1], height='auto')
#     ips = [input_image]
#     run_button.click(fn=process, inputs=ips, outputs=[result_gallery])

# block.launch(debug = True)

# block = gr.Blocks().queue()

gr.Markdown("## BRIA RMBG 1.4")
gr.HTML('''
  <p style="margin-bottom: 10px; font-size: 94%">
    This is a demo for BRIA RMBG 1.4 that using
    <a href="https://huggingface.co/briaai/RMBG-1.4" target="_blank">BRIA RMBG-1.4 image matting model</a> as backbone. 
  </p>
''')
title = "Background Removal"
description = r"""Background removal model developed by <a href='https://BRIA.AI' target='_blank'><b>BRIA.AI</b></a>, trained on a carefully selected dataset and is available as an open-source model for non-commercial use.<br> 
For test upload your image and wait. Read more at model card <a href='https://huggingface.co/briaai/RMBG-1.4' target='_blank'><b>briaai/RMBG-1.4</b></a>.<br>
"""
examples = [['./input.jpg'],]
# output = ImageSlider(position=0.5,label='Image without background', type="pil", show_download_button=True)
# demo = gr.Interface(fn=process,inputs="image", outputs=output, examples=examples, title=title, description=description)
demo = gr.Interface(fn=process,inputs="image", outputs="image", examples=examples, title=title, description=description)

if __name__ == "__main__":
    demo.launch(share=False)