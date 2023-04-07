from lib import *
from model import BiSeNetV2
from unet import UNet
from utils import *

# model = BiSeNetV2(3).eval().cuda()


model = UNet(3, 1).eval().cuda()
checkpoint = torch.load("unet/best_70.pth", map_location= 'cuda')["model_state_dict"]
model.load_state_dict(checkpoint)



normalize = transforms.Compose([
            transforms.ToTensor(),
        ])


img =  cv2.cvtColor(cv2.imread("538.png"), cv2.COLOR_BGR2RGB)

img = img[200:, :]
img = img/255
img = np.array(img, dtype=np.float32)
img = cv2.resize(img, (160, 80))
img = normalize(img)
model.eval()

start = time.time()

out = model(img.unsqueeze(0).cuda())
print(out.size())
out = out[0].squeeze(0)
print(out.size())
out = torch.where(out > 0.6, 255, 0)
cv2.imwrite('pred.png', out.cpu().numpy())
