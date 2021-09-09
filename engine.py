import os
import cv2
import torch
import time
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from tensorboardX import SummaryWriter
from visualize import visualize2
import numpy as np

class Tester(object):
    def __init__(self, model, weight_path):
        self.model = model
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            self.model = torch.nn.DataParallel(self.model,device_ids=range(torch.cuda.device_count()))
        self.model.to(device)
        
        data = torch.load(weight_path)
        print("Loading model epoch %d" % data['epoch'])
        self.model.load_state_dict(data['weight'])
    
    def test(self, data_loader):
        self.model.eval()
        matrixs = np.zeros([2,2],np.float32)
        with torch.no_grad():
            iteration_step = 0
            for image_A, image_B, change_mask_A, change_mask_B in tqdm(data_loader):
                image_A, image_B, change_mask_A, change_mask_B = image_A.cuda(), image_B.cuda(), change_mask_A.cuda(), change_mask_B.cuda()
                seg_A1, seg_B1, output_A = self.model(image_A, image_B)
                seg_B2, seg_A2, output_B = self.model(image_B, image_A)
                visualize(image_A, image_B, change_mask_A, output_A)
                visualize(image_B, image_A, change_mask_B, output_B)
                iteration_step += 1

                output_A = output_A.cpu().data.numpy().reshape(-1)
                output_B = output_B.cpu().data.numpy().reshape(-1)
                output_A[output_A >= 0.5] = 1
                output_A[output_A < 0.5] = 0
                output_B[output_B >= 0.5] = 1
                output_B[output_B < 0.5] = 0
                output_A = output_A.astype(np.int8)
                output_B = output_B.astype(np.int8)
                change_A = change_mask_A.cpu().data.numpy().reshape(-1).astype(np.int8)
                change_B = change_mask_B.cpu().data.numpy().reshape(-1).astype(np.int8)
                # test change mask A
                labels = list(set(np.concatenate((change_A, output_A), axis=0)))
                if (labels == [0]):
                    matrixs[0, 0] += confusion_matrix(change_A, output_A)[0, 0]
                elif (labels == [1]):
                    matrixs[1, 1] += confusion_matrix(change_A, output_A)[0, 0]
                else:
                    matrixs += confusion_matrix(change_A, output_A)

                # test change mask B
                labels = list(set(np.concatenate((change_B, output_B), axis=0)))
                if (labels == [0]):
                    matrixs[0, 0] += confusion_matrix(change_B, output_B)[0, 0]
                elif (labels == [1]):
                    matrixs[1, 1] += confusion_matrix(change_B, output_B)[0, 0]
                else:
                    matrixs += confusion_matrix(change_B, output_B)

        a, b, c, d = matrixs[0][0], matrixs[0][1], matrixs[1][0], matrixs[1][1]
        print(matrixs)
        accuracy = (a + d) / (a + b + c + d)
        if((d+c)!= 0):
            recall = d / (d + c)
        else:
            recall = 0
        if ((d + b) != 0):
            precision = d / (d + b)
        else:
            precision = 0

        F1 = 2 * d / (a + b + c + d + d - a)
        if ((d + b + c) != 0):
            iou = d / (c + b + d)
        else:
            iou = 0

        print("Accuracy = %g, recall = %g, precision = %g, F1 = %g, iou = %g" % (accuracy, recall, precision, F1, iou))

class Trainer(object):
    def __init__(self, model, optimizer, loss_f, save_dir=None, save_freq=1):
        self.model = model
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            self.model = torch.nn.DataParallel(self.model,device_ids=range(torch.cuda.device_count()))
        self.model.to(device)
        #self.model.load_state_dict(torch.load("checkpoints/sexp6_DA_4/ model_400.pkl")['weight'])
        self.optimizer = optimizer
        self.loss_f = loss_f().cuda()
        self.a_loss_f = torch.nn.BCELoss().cuda()
        self.save_dir = save_dir
        self.save_freq = save_freq
        self.writer = SummaryWriter(save_dir)

    def test(self, data_loader, epoch):
        self.model.eval()
        matrixs = np.zeros([2,2],np.float32)
        with torch.no_grad():
            iteration_step = 0
            for image_A, image_B, change_mask_A, change_mask_B in tqdm(data_loader):
                image_A, image_B, change_mask_A, change_mask_B = image_A.cuda(), image_B.cuda(), change_mask_A.cuda(), change_mask_B.cuda()
                seg_A1, seg_B1, output_A = self.model(image_A, image_B)
                seg_B2, seg_A2, output_B = self.model(image_B, image_A)
                print(np.max(seg_A1.cpu().numpy()))
                print(np.max(seg_A2.cpu().numpy()))
                print(np.max(seg_B1.cpu().numpy()))
                print(np.max(seg_B2.cpu().numpy()))
                print(np.max(output_A.cpu().numpy()))
                print(np.max(output_B.cpu().numpy()))
                iteration_step += 1

                output_A = output_A.cpu().data.numpy().reshape(-1)
                output_B = output_B.cpu().data.numpy().reshape(-1)
                output_A[output_A >= 0.5] = 1
                output_A[output_A < 0.5] = 0
                output_B[output_B >= 0.5] = 1
                output_B[output_B < 0.5] = 0
                output_A = output_A.astype(np.int8)
                output_B = output_B.astype(np.int8)
                change_A = change_mask_A.cpu().data.numpy().reshape(-1).astype(np.int8)
                change_B = change_mask_B.cpu().data.numpy().reshape(-1).astype(np.int8)
                # test change mask A
                labels = list(set(np.concatenate((change_A, output_A), axis=0)))
                if (labels == [0]):
                    matrixs[0, 0] += confusion_matrix(change_A, output_A)[0, 0]
                elif (labels == [1]):
                    matrixs[1, 1] += confusion_matrix(change_A, output_A)[0, 0]
                else:
                    matrixs += confusion_matrix(change_A, output_A)

                # test change mask B
                labels = list(set(np.concatenate((change_B, output_B), axis=0)))
                if (labels == [0]):
                    matrixs[0, 0] += confusion_matrix(change_B, output_B)[0, 0]
                elif (labels == [1]):
                    matrixs[1, 1] += confusion_matrix(change_B, output_B)[0, 0]
                else:
                    matrixs += confusion_matrix(change_B, output_B)

        a, b, c, d = matrixs[0][0], matrixs[0][1], matrixs[1][0], matrixs[1][1]
        print(matrixs)
        accuracy = (a + d) / (a + b + c + d)
        if((d+c)!= 0):
            recall = d / (d + c)
        else:
            recall = 0
        if ((d + b) != 0):
            precision = d / (d + b)
        else:
            precision = 0

        F1 = 2 * d / (a + b + c + d + d - a)
        if ((d + b + c) != 0):
            iou = d / (c + b + d)
        else:
            iou = 0

        print("Epoch : %d" % epoch)
        print("Accuracy = %g, recall = %g, precision = %g, F1 = %g, iou = %g" % (accuracy, recall, precision, F1, iou))

        self.writer.add_scalar('test/accuracy', accuracy, epoch)
        self.writer.add_scalar('test/recall', recall, epoch)
        self.writer.add_scalar('test/precision', precision, epoch)
        self.writer.add_scalar('test/F1', F1, epoch)
        self.writer.add_scalar('test/iou', iou, epoch)
        return 

    def train(self, data_loader, epoch):
        self.model.train()
        loop_loss = []
        with torch.enable_grad():
            iteration_step = 0
            for image_A, image_B, mask_A, mask_B, change_mask in data_loader:
                # DEBUG
                # visualize2(image_A, image_B, mask_A, mask_B, change_mask, name)
                image_A, image_B, mask_A, mask_B, change_mask = image_A.cuda(), image_B.cuda(), mask_A.cuda(), mask_B.cuda(), change_mask.cuda()
                self.optimizer.zero_grad()
                seg_A, seg_B, output = self.model(image_A, image_B)
                loss = 0.5*self.loss_f(output, change_mask) + 0.25 * self.a_loss_f(seg_A, mask_A) + 0.25*self.a_loss_f(seg_B, mask_B)
                loss_step = loss.data.item()
                if(iteration_step % 20 == 0):
                    print("%s Iteration %d loss = %g" % (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), iteration_step, loss_step))
                iteration_step += 1
                loop_loss.append(loss.data.item() / len(data_loader))

                loss.backward()
                self.optimizer.step()

        self.writer.add_scalar('train/loss_epoch', sum(loop_loss), epoch)
        print(">>>[{mode}] loss: {loss}".format(mode='train',loss=sum(loop_loss)))
        return loop_loss

    def loop(self, epochs, train_data, test_data, scheduler=None):
        for ep in range(1, epochs + 1):
            if scheduler is not None:
                scheduler.step()
            print("epochs: {}".format(ep))
            self.train(train_data,ep)
            if(ep % 5 == 0):
                self.test(test_data,ep)
            if (ep % self.save_freq == 0):
                self.save(ep)

    def save(self, epoch, **kwargs):
        model_out_path = self.save_dir
        state = {"epoch": epoch, "weight": self.model.state_dict()}
        if not os.path.exists(model_out_path):
            os.makedirs(model_out_path)
        torch.save(state, model_out_path + '/model_{epoch}.pkl'.format(epoch=epoch))
