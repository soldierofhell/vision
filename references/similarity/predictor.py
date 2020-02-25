import torch
import torchvision.transforms as transforms

from .model import EmbeddingNet

class Predictor:

  def __init__(self, checkpoint_path):

    self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    self.model = EmbeddingNet()
    self.model.load_state_dict(torch.load(checkpoint_path))

    self.model.to(self.device)
    self.model.eval()
    
    self.transform = transforms.Compose([transforms.Lambda(lambda image: image.convert('RGB')),
                                transforms.Resize((224, 224)),
                                transforms.ToTensor()])

  def _preprocess(self, image):
    image = transforms.ToPILImage()(image)
    image = self.transform(image)
    
    return image

  
  def predict(self, image_list):
    
    image_tensor = torch.cat([self._preprocess(im).unsqueeze(0) for im in image_list], dim=0)

    with torch.no_grad():
      image_tensor = image_tensor.cuda()      
      embedings = model(image_tensor)
      
    return embedings.cpu().numpy()
      
      # todo: obliczenia wylacznie na cuda
      # 1. preprocessing bez PIL
      # 2. torch.cdist()
      
      
class TensorPredictor:

  def __init__(self, checkpoint_path):

    self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    self.model = EmbeddingNet()
    self.model.load_state_dict(torch.load(checkpoint_path))

    self.model.to(self.device)
    self.model.eval()    

  def predict(self, image_list):
    
    tensor_list = []
    
    for image_tensor in image_list:
      image_tensor = image_tensor[:,:,::-1]  # BGR -> RBG ?
      image_tensor = F.interpolate(image_tensor, size=(224, 224))
      tensor_list.append(image_tensor)
    
    input_tensor = torch.stack(tensor_list)

    with torch.no_grad():
      input_tensor = input_tensor.to(self.device)      
      embedings = model(input_tensor)
      
    return embedings
