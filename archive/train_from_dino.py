from training_loops.trainFuncs import *
import tifffile
import torchvision.transforms as T
from logging_utils import logger


def load_image(img: str) -> torch.Tensor:
    """
    Load an image and return a tensor that can be used as an input to DINOv2.
    """
    # img = Image.open(img)
    tif_image = tifffile.imread(img)[1:99, 1:99, 1:4]
    numpy_image = np.array(tif_image).astype(np.float32)
    reshaped_image = np.transpose(numpy_image, (2, 0, 1))
    tensor_image = torch.tensor(reshaped_image, dtype=torch.float32)
    final_image = torch.unsqueeze(tensor_image, 0)
    return final_image

    transform_image = T.Compose([T.ToTensor(), T.Resize(244), T.CenterCrop(224), T.Normalize([0.5], [0.5])])
    transformed_img = transform_image(img)[:3].unsqueeze(0)
    return transformed_img


logger.info('---Start Training---')
reg_config = parse_yaml('../configs/regBasicDINOv2.yaml')
log_dir = reg_config['logging_params']['save_dir']


dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
dinov2_vits14.to(device)

file = '/home/sam/Desktop/so2sat_test/So2Sat_POP_Part1/test/00331_204371_munich/sen2summer/Class_14/1kmN2775E4433_sen2summer.tif'
embeddings = dinov2_vits14(load_image(file).to(device))
logger.info(embeddings.shape)


logger.info('---Training Regression model---')
train_reg_from_dino(reg_config, log_dir, dinov2_vits14)
