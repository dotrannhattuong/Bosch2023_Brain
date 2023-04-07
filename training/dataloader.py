from lib import *
from utils import *

class CDSDataset(Dataset):

    color_encoding = [
        ('sky', (128, 128, 128)),
        ('road', (128, 0, 0)),
        ('car', (192, 192, 128)),
    ]


    def __init__(self, mode='train', num_classes=3):
        self.mode = mode
        self.num_classes = num_classes
        # Normailization 
        self.normalize = transforms.Compose([
            transforms.ToTensor()
        ])
        self.data_files = sorted(glob('segment/imgs/*'))
        self.label_files = sorted(glob('segment/masks/*'))
        # self.DATA_PATH = os.path.join(os.getcwd(), 'data')
    
        # # print( self.DATA_PATH)
        # self.train_path, self.val_path = [os.path.join(self.DATA_PATH, x) for x in ['train', 'val']]
        
        # if self.mode == 'train':
        #     self.data_files = self.get_files(self.train_path)
        #     self.label_files = [self.get_label_file(f, 'train', 'trainanot') for f in self.data_files]
        # elif self.mode == 'val':
        #     self.data_files = self.get_files(self.val_path)
        #     self.label_files = [self.get_label_file(f, 'val', 'valanot') for f in self.data_files]


        # else:
        #     raise RuntimeError("Unexpected dataset mode. "
        #                        "Supported modes are: train, val and test")
    
    def get_files(self, data_folder):
        """
            Return all files in folder with extension 
        """
        return glob("{}/*.{}".format(data_folder, 'png'))
    
    def get_label_file(self, data_path, data_dir, label_dir):
        """
            Return label path for data_path file 
        """
        data_path = data_path.replace(data_dir, label_dir)
        fname, ext = data_path.split('.')
        return "{}.{}".format(fname, ext)


    def image_loader(self, data_path, label_path):
        data =  cv2.cvtColor(cv2.imread(data_path), cv2.COLOR_BGR2RGB)
        data = data[200:, :]
        data = data/255
        data = cv2.resize(data, (160, 80))
        # label = Image.open(label_path)

        mask = cv2.imread(label_path, 0)
        mask = mask[200:, :]
        # mask = mask / 255
        # mask = np.where(mask == 75, 2, mask)
        # mask = np.where(mask == 38, 1, mask)
        mask = np.where(mask == 75, 0, mask)
        mask = np.where(mask == 38, 1, mask)
        mask = cv2.resize(mask, (160, 80))
        # mask = np.bincount(mask.flatten())
        # print(mask)
        # exit()
        # onehot = np.eye(3)[mask]
        return data, mask.reshape(1, 80, 160)#onehot.transpose(2, 0, 1)
            
    
    def __getitem__(self, index):

        
        data_path, label_path = self.data_files[index], self.label_files[index]
        img, label = self.image_loader(data_path, label_path)

        # Apply normalization in img
        # augment_hsv(img, hgain=0.015, sgain=0.7, vgain=0.4)
        img = self.normalize(img)
        img = torch.tensor(img, dtype=torch.float)
        label = np.array(label)
        label = torch.from_numpy(label).float()
        # print(img.size(), label.size())
        return img, label 
    
    def __len__(self):
        """
            Return len of dataset 
        """
        return len(self.data_files)