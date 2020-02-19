import torch
import torchvision.transforms as transforms

from model import EmbeddingNet

class Predictor:

  def __init__(self, checkpoint_path, threshold):

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    self.model = EmbeddingNet()
    self.model.load_state_dict(torch.load(checkpoint_path))

    self.model.to(device)
    self.model.eval()
    
    self.transform = transforms.Compose([transforms.Lambda(lambda image: image.convert('RGB')),
                                transforms.Resize((224, 224)),
                                transforms.ToTensor()])



  def predict(self, image): 

    with torch.no_grad():
    
      image = transforms.ToPILImage()(image).convert("L")
      image = self.transform(image)
      image = image.cuda()
      
      out = model(image)
