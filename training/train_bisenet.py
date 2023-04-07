from lib import *
from model import BiSeNetV2
from dataloader import CDSDataset
from utils import val, DiceLoss

# Training 
EPOCHS = 300
LEARNING_RATE = 0.001
BATCH_SIZE = 8
CHECKPOINT_STEP = 10
VALIDATE_STEP = 5
NUM_CLASSES = 3
max_miou = 0
loss_plt = []
miou_plt = []
val_loss_plt = []
main_loss_plt = []


# model = UNet(3)


dice_loss = DiceLoss()
loss_func = nn.CrossEntropyLoss()
model = BiSeNetV2(3).cuda()
# Dataloader for train
dataset_train = CDSDataset(mode='train')
dataloader_train = DataLoader(
    dataset_train,
    batch_size=BATCH_SIZE,
    shuffle=True
)

# Dataloader for validate
dataset_val = CDSDataset(mode='val')
dataloader_val = DataLoader(
    dataset_val,
    batch_size=1,
    shuffle=True
)

# Optimizer 
torch.backends.cudnn.benchmark = True # new
optimizer = torch.optim.Adam(model.parameters(), LEARNING_RATE)

#new
def lambda_epoch(epoch): 
    return math.pow(1-epoch/EPOCHS, 0.9)

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_epoch, last_epoch=-1)
for epoch in range(1, EPOCHS):
    model.train()
    tq = tqdm(total=len(dataloader_train) * BATCH_SIZE)
    tq.set_description('Epoch {}/{}'.format(epoch, EPOCHS))

    loss_record = []
    main_loss_record = []
    for i, (data, label) in enumerate(dataloader_train):
        data = data.cuda()
        label = label.cuda()

        output = model(data) #output_shape ([4, 12, 720, 960])
    
        loss = loss_func(torch.sigmoid(output[0]), label) + loss_func(torch.sigmoid(output[1]), label) + loss_func(torch.sigmoid(output[2]), label) + loss_func(torch.sigmoid(output[3]), label) + loss_func(torch.sigmoid(output[4]), label)#\
              #+ dice_loss(torch.sigmoid(output[0]), label) + dice_loss(torch.softmax(output[1], dim = 1), label) + dice_loss(torch.softmax(output[2], dim = 1), label) + dice_loss(torch.softmax(output[3], dim = 1), label) + dice_loss(torch.softmax(output[4], dim = 1), label)
        tq.update(BATCH_SIZE)
        tq.set_postfix(loss='%.6f' % loss)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss_record.append(loss.item())
        
    scheduler.step()
    tq.close()

    loss_train_mean = np.mean(loss_record)
    
    print("lr: %f" % optimizer.param_groups[0]['lr'])
    print('loss for train : %f' % (loss_train_mean))
    
    # Save checkpoint 
    # if epoch % CHECKPOINT_STEP == 0:
        # name = os.path.join(os.getcwd(), "last_" + str(epoch) + ".pth")
        # torch.save({
        #       'model_state_dict': model.state_dict(),
        #       }, name)
        
    # Validate save best model 
    # Save checkpoint 
    if epoch % VALIDATE_STEP == 0:
        val_time_start = time.time()
        mean_iou = val(model, dataloader_val)
        val_time_end = time.time()
        print("val_time: %f" % (val_time_end - val_time_start))
        
        if mean_iou > max_miou:
            max_miou = mean_iou
            print('Save best model with mIoU = {}'.format(mean_iou))
            # torch.save(model.state_dict(), 'checkpoints/best_model.pth')
            name = os.path.join(os.getcwd(), "weight_sig_2/best_" + str(epoch) + ".pth")
            torch.save({
              'model_state_dict': model.state_dict(),
#               'optimizer_state_dict': optimizer.state_dict(),
              }, name)
            