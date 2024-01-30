import numpy as np
import torch.nn.functional as F
from torchvision.transforms.functional import normalize
from foo import hello
import gradio as gr
import git  # pip install gitpython

hello()

git.Git(".").clone("https://huggingface.co/briaai/RMBG-1.4")
# git.Git(".").clone("git@hf.co:briaai/RMBG-1.4")
from briarmbg import BriaRMBG

net=BriaRMBG()
model_path = "./model.pth"
if torch.cuda.is_available():
    net.load_state_dict(torch.load(model_path))
    net=net.cuda()
else:
    net.load_state_dict(torch.load(model_path,map_location="cpu"))
net.eval() 

def image_size_by_min_resolution(
    image: Image.Image,
    resolution: Tuple,
    resample=None,
):
    w, h = image.size  

    image_min = min(w, h)
    resolution_min = min(resolution)

    scale_factor = image_min / resolution_min

    resize_to: Tuple[int, int] = (
        int(w // scale_factor),
        int(h // scale_factor),
    )
    return resize_to
    

def resize_image(image):
    image = image.convert('RGB')
    new_image_size = image_size_by_min_resolution(image=image,resolution=(1024, 1024))
    image = image.resize(new_image_size, Image.BILINEAR)
    return image


def process(input_image):

    # prepare input
    orig_image = Image.open(im_path)
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

    # save result
    im_array = (result*255).cpu().data.numpy().astype(np.uint8)
    pil_im = Image.fromarray(np.squeeze(im_array))
    # paste the mask on the original image
    new_im = Image.new("RGBA", pil_im.size, (0,0,0))
    new_im.paste(orig_image, mask=pil_im)

    return new_im


block = gr.Blocks().queue()

with block:
    gr.Markdown("## BRIA RMBG 1.4")
    gr.HTML('''
      <p style="margin-bottom: 10px; font-size: 94%">
        This is a demo for BRIA RMBG 1.4 that using
        <a href="https://huggingface.co/briaai/RMBG-1.4" target="_blank">BRIA RMBG-1.4 image matting model</a> as backbone. 
      </p>
    ''')
    with gr.Row():
        with gr.Column():
            # input_image = gr.Image(sources=None, type="pil") # None for upload, ctrl+v and webcam
            input_image = gr.Image(sources=None, type="numpy") # None for upload, ctrl+v and webcam
            run_button = gr.Button(value="Run")
            
        with gr.Column():
            result_gallery = gr.Gallery(label='Output', show_label=False, elem_id="gallery", columns=[2], height='auto')
    ips = [input_image]
    run_button.click(fn=process, inputs=ips, outputs=[result_gallery])

block.launch(debug = True)