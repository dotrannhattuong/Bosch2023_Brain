from lib import *
NUM_CLASSES = 1
def augment_hsv(im, hgain= 0, sgain=0.5, vgain=0.5):
    # HSV color-space augmentation
    if hgain or sgain or vgain:
        r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
        hue, sat, val = cv2.split(cv2.cvtColor(im, cv2.COLOR_BGR2HSV))
        dtype = im.dtype  # uint8

        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR, dst=im)  # no return needed
  
def reverse_one_hot(image):
    # Convert output of model to predicted class 
    image = image.permute(1, 2, 0)
    x = torch.argmax(image, dim=-1)
    return x


def compute_accuracy(pred, label):
    pred = pred.flatten()
    label = label.flatten()
    total = len(label)
    count = 0.0
    for i in range(total):
        if pred[i] == label[i]:
            count = count + 1.0
    return float(count) / float(total)



def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return torch.bincount(n * a[k].type(torch.int) + b[k], minlength=n ** 2).reshape(n, n)


def per_class_iu(hist):
    epsilon = 1e-5
    return (torch.diag(hist) + epsilon) / (hist.sum(1) + hist.sum(0) - torch.diag(hist) + epsilon)

def val(model, dataloader):
    accuracy_arr = []
    
    hist = torch.zeros((NUM_CLASSES, NUM_CLASSES)).cuda()
    with torch.no_grad():
        miou_list = []
        model.eval()
        model = model.cuda()
        print('Starting validate')
        loss_record = []
        for i, (val_data, val_label) in enumerate(dataloader):
            val_data = val_data.cuda()
            val_label = val_label.cuda()
            # The output of model is (1, num_classes, W, H) => (num_classes, W, H)
            val_output = model(val_data)
            # val_output = torch.sigmoid(val_output)
            val_output = val_output[0].squeeze()

            # Convert the (num_classes, W, H) => (W, H) with one hot decoder 
            # val_output = reverse_one_hot(val_output)
            
            # Process label. Convert to (W, H) image 
            val_label = val_label.squeeze()
            val_output = torch.where(val_output > 0.5, torch.tensor([1], device='cuda'), torch.tensor([0], device='cuda'))
            # print(val_output.get_device(), val_label.get_device())
            # hist += fast_hist(val_label.flatten().long(), val_output.flatten().long(), NUM_CLASSES)

            itersect = torch.sum(val_output * val_label)
            union = torch.sum(val_output) + torch.sum(val_label) - itersect
            iou = itersect/union
            miou_list.append(iou.item())

        # miou_list = per_class_iu(hist)[:-1]
        mean_iou =  np.mean(miou_list)
        print('Mean IoU: {}'.format(mean_iou))
        return mean_iou

def colour_code_segmentation(image, label_values):
    w = image.shape[0]
    h = image.shape[1]
    x = np.zeros([w,h,3], dtype=np.uint8)
    
    colour_codes = label_values
    for i in range(0, w):
        for j in range(0, h):  
            index = int(image[i, j])
            if index == 2:
                index = 1
            x[i, j, :] = colour_codes[index]
    return x

color_encoding = [
        ('sky', (128, 128, 128)),
        ('road', (128, 0, 0)),
        ('car', (192, 192, 128)),
    ]
label_values = [v[1]for v in color_encoding]
def img_show(img):
    plt.imshow(img, interpolation='nearest')
    plt.show()
    
def to_rgb(image):
    test_label_vis = colour_code_segmentation(image, label_values)
    img = Image.fromarray(test_label_vis, 'RGB')
    return img
def save_image(output, gt, img):
    cv2.imwrite("output.jpg",  cv2.hconcat([np.array(gt), cv2.hconcat([np.array(img), np.array(output)])]))


class DiceLoss(nn.Module):
    """
    Examples:
        >>> N = 5  # num_classes
        >>> loss = tgm.losses.DiceLoss()
        >>> input = torch.randn(1, N, 3, 5, requires_grad=True)
        >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        >>> output = loss(input, target)
        >>> output.backward()
    """

    def __init__(self) -> None:
        super(DiceLoss, self).__init__()
        self.eps: float = 1e-6

    def _one_hot_encoder(self, input_tensor, num_classes):
        tensor_list = []
        for i in range(num_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def forward(
            self,
            input: torch.Tensor,
            target: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(input):
            raise TypeError("Input type is not a torch.Tensor. Got {}"
                            .format(type(input)))
        if not len(input.shape) == 4:
            raise ValueError("Invalid input shape, we expect BxNxHxW. Got: {}"
                             .format(input.shape))
        if not input.shape[-2:] == target.shape[-2:]:
            raise ValueError("input and target shapes must be the same. Got: {}"
                             .format(input.shape, input.shape))
        if not input.device == target.device:
            raise ValueError(
                "input and target must be in the same device. Got: {}" .format(
                    input.device, target.device))
        # compute softmax over the classes axis
        input_soft = F.softmax(input, dim=1)

        # create the labels one hot tensor
        target_one_hot = self._one_hot_encoder(target, num_classes=input.shape[1])
        # compute the actual dice score
        dims = (1, 2, 3)
        intersection = torch.sum(input_soft * target_one_hot, dims)
        cardinality = torch.sum(input_soft + target_one_hot, dims)

        dice_score = 2. * intersection / (cardinality + self.eps)
        return torch.mean(1. - dice_score)