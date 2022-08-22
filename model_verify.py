from mmseg.apis import inference_segmentor, init_segmentor
import mmcv

config_file = 'configs/upernet/upernet_r101_512x1024_40k_cityscapes.py'
checkpoint_file = 'upernet_r50_512x1024_40k_cityscapes_20200605_094827-aa54cb54.pth'

# build the model from a config file and a checkpoint file
model = init_segmentor(config_file, checkpoint_file, device='cuda:0')

# test a single image and show the results
img = 'demo.png' # or img = mmcv.imread(img) #which will only load it once
result = inference_segmentor(model, img)
# save the visualization results to image files
model.show_result(img, result, out_file='result.jpg')
print("model and checkpoint file works, you can download result.jpg file to verify results!")
