!pip3 install transformers

import os
import torch
#import logging
#import random
import numpy as np
import pandas as pd
import torch.nn as nn


from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset,DataLoader
#from keras.preprocessing.sequence import pad_sequences

batchSize = 16
learningRate = 0.03 #0.03 = 0.9134166 6
numEpoch = 5
device = 'cuda'
torch.cuda.empty_cache()

class MyDataset(Dataset):
    def __init__(self, data, label, isTest):
      self.data = data
      self.label = label
      self.isTest = isTest

    def __getitem__(self,index):
      #data = {'input_ids': self.data['input_ids'][index], 'token_type_ids': self.data['token_type_ids'][index], 'attention_mask': self.data['attention_mask'][index]}
      #print(self.data['input_ids'][index], self.data['token_type_ids'][index], self.data['attention_mask'][index])
      if self.isTest:
        return self.data[index]
      else:
        return self.data[index], self.label[index]

    def __len__(self):
      return len(self.data)

class MyModel(torch.nn.Module):
    def __init__(self, bert):
        super(MyModel, self).__init__()
        self.bert = bert
        self.linear = nn.Linear(768, 384)
        #self.linear2 = nn.Linear(768, 768)
        self.linear1 = nn.Linear(384, 3)
        self.linear2 = nn.Linear(384, 3)
        self.linear3 = nn.Linear(384, 3)
        self.linear4 = nn.Linear(384, 3)
        self.linear5 = nn.Linear(384, 3)
        self.linear6 = nn.Linear(384, 3)
        self.linear7 = nn.Linear(384, 3)
        self.linear8 = nn.Linear(384, 3)
        self.linear9 = nn.Linear(384, 3)
        self.linear10 = nn.Linear(384, 3)
        self.linear11 = nn.Linear(384, 3)
        self.linear12 = nn.Linear(384, 3)
        self.linear13 = nn.Linear(384, 3)
        self.linear14 = nn.Linear(384, 3)
        self.linear15 = nn.Linear(384, 3)
        self.linear16 = nn.Linear(384, 3)
        self.linear17 = nn.Linear(384, 3)
        self.linear18 = nn.Linear(384, 3)
        #self.acf = nn.Sigmoid()

    def forward(self, batch):
        #print('this is forward', batch)
        input = {'input_ids': batch[:, :, 0], 'token_type_ids': batch[:, :, 1], 'attention_mask': batch[:, :, 2]}
        output1 = self.bert(**input)

        clsEmbed = output1[1][:][:]
        output = self.linear(clsEmbed)
        #output = self.linear2(output)
        #finalOutput = torch.tensor([]).to(device)
        for i in range(18):
            if i == 0:
                o1 = self.linear1(output)
            elif i == 1:
                o2 = self.linear2(output)
            elif i == 2:
                o3 = self.linear3(output)
            elif i == 3:
                o4 = self.linear4(output)
            elif i == 4:
                o5 = self.linear5(output)
            elif i == 5:
                o6 = self.linear6(output)
            elif i == 6:
                o7 = self.linear7(output)
            elif i == 7:
                o8 = self.linear8(output)
            elif i == 8:
                o9 = self.linear9(output)
            elif i == 9:
                o10 = self.linear10(output)
            elif i == 10:
                o11 = self.linear11(output)
            elif i == 11:
                o12 = self.linear12(output)
            elif i == 12:
                o13 = self.linear13(output)
            elif i == 13:
                o14 = self.linear14(output)
            elif i == 14:
                o15 = self.linear15(output)
            elif i == 15:
                o16 = self.linear16(output)
            elif i == 16:
                o17 = self.linear17(output)
            elif i == 17:
                o18 = self.linear18(output)
        finalOutput = torch.stack((o1, o2, o3, o4, o5, o6, o7, o8, o9, o10, o11, o12, o13, o14, o15, o16, o17, o18), 2)
            #finalOutput = torch.permute(finalOutput, (1, 0, 2))

            #if i == 0:
            #    finalOutput = self.linear3[i](output)
            #else:
            #    torch.stack((finalOutput, self.linear3[i](output)))
        #finalOutput = self.acf(finalOutput)
          #if i == 0:
            #output = self.linear[i](clsEmbed).unsqueeze(2)
          #else:
            #output = torch.stack((output, self.linear[i](clsEmbed).unsqueeze(2)), dim=2)
        return finalOutput

#def collate_fn(batch):
#  batch['input_ids'] = batch[]
#  return batch

def main():


  # Tokenizer and Bert Model
  tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
  embedding = AutoModel.from_pretrained('bert-base-chinese')


  train_data = pd.read_csv("../input/nlpfinal/train.csv")
  dev_data = pd.read_csv("../input/nlpfinal/dev.csv")
  test_data = pd.read_csv("../input/nlpfinal/test.csv")


  #trainList = train_data.loc[:, 'review'].tolist()
  #devList = dev_data.loc[:, 'review'].tolist()
  #testList = test_data.loc[:, 'review'].tolist()
  #print(trainList)

  train_token = tokenizer(train_data.loc[:, 'review'].tolist(), padding=True, truncation=True, return_tensors="pt")
  train_token = torch.permute(torch.stack((train_token['input_ids'], train_token['token_type_ids'], train_token['attention_mask'])), (1, 2, 0))
  #train_token['input_ids'] = train_token['input_ids'].to(device)
  dev_token = tokenizer(dev_data.loc[:, 'review'].tolist(), padding=True, truncation=True, return_tensors="pt")
  dev_token = torch.permute(torch.stack((dev_token['input_ids'], dev_token['token_type_ids'], dev_token['attention_mask'])), (1, 2, 0))
  #dev_token['input_ids'] = dev_token['input_ids'].to(device)
  test_token = tokenizer(test_data.loc[:, 'review'].tolist(), padding=True, truncation=True, return_tensors="pt")
  test_token = torch.permute(torch.stack((test_token['input_ids'], test_token['token_type_ids'], test_token['attention_mask'])), (1, 2, 0))
  #test_token['input_ids'] = test_token['input_ids'].to(device)
  #print(train_token.shape, test_token.shape, dev_token.shape)

  train_label, dev_label = [], []
  for i in range(2, 20):
    tmp1 = train_data.iloc[:, i].tolist()
    for j in range(len(tmp1)):
        if tmp1[j] == -2:
            tmp1[j] = 3
        else:
            tmp1[j] += 1
    train_label.append(tmp1)
    
    tmp2 = dev_data.iloc[:, i].tolist()
    for j in range(len(tmp2)):
        if tmp2[j] == -2:
            tmp2[j] = 3
        else:
            tmp2[j] += 1
    dev_label.append(tmp2)
  
  train_label = np.array(train_label)
  train_label = train_label.T
  train_label = torch.from_numpy(train_label)
  dev_label = np.array(dev_label)
  dev_label = dev_label.T
  dev_label = torch.from_numpy(dev_label)

  #print(len(train_token), len(train_label))
  train_dataset = MyDataset(train_token, train_label, False)
  dev_dataset = MyDataset(dev_token, dev_label, False)
  test_dataset = MyDataset(test_token, 0, True)

  train_loader = DataLoader(dataset=train_dataset, batch_size=batchSize, shuffle=True)
  dev_loader = DataLoader(dataset=dev_dataset, batch_size=batchSize, shuffle=False)
  test_loader = DataLoader(dataset=test_dataset, batch_size=batchSize, shuffle=False)

  model = MyModel(embedding).to(device)

  optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)
  criterion = nn.CrossEntropyLoss(ignore_index=3).to(device)


  highest_acc = 0
  step = 0
  for epoch in range(numEpoch):
    print('epochStart')
    model.train() # set the model to training mode
    for i, data in enumerate(train_loader):
      inputs, labels = data[:][0], data[:][1]
      if torch.cuda.is_available():
        inputs, labels = inputs.to(device), labels.to(device)
      #print(inputs)
      optimizer.zero_grad()
      outputs = model(inputs)
      #outputs = outputs.permute(2, 1, 0)
      
      #print(outputs.shape, labels.shape)
      batch_loss = criterion(outputs, labels)
      batch_loss.backward()
      optimizer.step()
      print(step)
      step += 1
      #print(outputs.shape, outputs)
      #print(labels)


    val_acc = 0.0
    val_loss = 0.0
    val_useless = 0
    model.eval() # set the model to evaluation mode
    with torch.no_grad():
        for i, data in enumerate(dev_loader):
            inputs, labels = data[:][0], data[:][1]
            if torch.cuda.is_available():
              inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            batch_loss = criterion(outputs, labels)

            _, val_pred = torch.max(outputs, 1)
            for j in range(len(labels)):
              for s in range(18):
                if val_pred[j][s] == labels[j][s]:
                  val_acc += 1
                if labels[j][s] == 3:
                  val_useless += 1
            val_loss += batch_loss.item()

        val_acc = val_acc / (18 * len(dev_label) - val_useless)
        print('val_loss :', val_loss)
        print('val_acc :', val_acc)

        if val_acc > highest_acc:
            highest_acc = val_acc
            torch.save(model.state_dict(), "model.pt")
            print('saving model with acc {:.3f}'.format(val_acc))
        


  
  #testModel = MyModel(embedding).to(device)
  model.eval()
  ckpt = torch.load("model.pt")
  # load weights into model
  model.load_state_dict(ckpt)

  prediction = []
  with torch.no_grad():
      for i, data in enumerate(test_loader):
          if torch.cuda.is_available():
            data = data.to(device)
          outputs = model(data)
          _, val_pred = torch.max(outputs, 1)
          val_pred = val_pred.tolist()

          for j in range(len(val_pred)):
              prediction.append(val_pred[j])
          #for j in range(len(outputs)):
          #    cur_pred = []
          #    for s in range(18):
          #        if outputs[j][s] >= 0.5:
          #          cur_pred.append(1)
          #        else:
          #          cur_pred.append(0)
          #    prediction.append(cur_pred)
            
  
  testList = test_data.loc[:, 'id'].tolist()
  #print('pred len =', len(prediction), 'test len =', len(testList))
  ids = [str(i)+'-'+str(j) for i in testList for j in range(1, 19)]
  predicted = [prediction[i][j] for i in range(len(prediction)) for j in range(18)]
  predicted = np.array(predicted)
  predicted = predicted - 1
  #print(predicted, ids, len(predicted), len(ids))
  outDataFrame = pd.DataFrame({'id-#aspect': ids, 'sentiment': predicted})
  outDataFrame.to_csv('task2.csv', index=False)

    # TODO: write prediction to file (args.pred_file)
  #with open(args.pred_file, 'w') as f:
      #f.write('id,intent\n')
      #for i, y in enumerate(prediction):
          #f.write('{},{}\n'.format(dataset[i]['id'], dataset.idx2label(y)))

if __name__ == '__main__':
  main()
