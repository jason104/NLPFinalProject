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

batchSize = 128
learningRate = 0.0001
numEpoch = 10
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
        self.linear = nn.Linear(768, 18)
        self.acf = nn.Sigmoid()

    def forward(self, batch):
        #print('this is forward', batch)
        input = {'input_ids': batch[:, :, 0], 'token_type_ids': batch[:, :, 1], 'attention_mask': batch[:, :, 2]}
        output1 = self.bert(**input)

        clsEmbed = output1[1][:][:]
        output = self.linear(clsEmbed)
        output = self.acf(output)
          #if i == 0:
            #output = self.linear[i](clsEmbed).unsqueeze(2)
          #else:
            #output = torch.stack((output, self.linear[i](clsEmbed).unsqueeze(2)), dim=2)
        return output

#def collate_fn(batch):
#  batch['input_ids'] = batch[]
#  return batch

def main():


  # Tokenizer and Bert Model
  tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
  embedding = AutoModel.from_pretrained('bert-base-chinese')


  train_data = pd.read_csv("train.csv")
  dev_data = pd.read_csv("dev.csv")
  test_data = pd.read_csv("test.csv")


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
    for label in tmp1:
      if label == -2:
        label = 0
      else:
        label = 1
    train_label.append(tmp1)
    tmp2 = dev_data.iloc[:, i].tolist()
    for label in tmp2:
      if label == -2:
        label = 0
      else:
        label = 1
    dev_label.append(tmp2)
  
  train_label = np.array(train_label, dtype='float32')
  train_label = train_label.T
  train_label = torch.from_numpy(train_label)
  dev_label = np.array(dev_label, dtype='float32')
  dev_label = dev_label.T
  dev_label = torch.from_numpy(dev_label)

  #print(len(train_token), len(train_label))
  train_dataset = MyDataset(train_token, train_label, False)
  dev_dataset = MyDataset(dev_token, dev_label, False)
  test_dataset = MyDataset(test_token, 0, True)

  train_loader = DataLoader(dataset=train_dataset, batch_size=batchSize, shuffle=True)
  dev_loader = DataLoader(dataset=dev_dataset, batch_size=batchSize, shuffle=False)
  test_loader = DataLoader(dataset=test_dataset, batch_size=batchSize, shuffle=False)

  model = MyModel(embedding)

  optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)
  criterion = nn.BCELoss()


  lowest_loss = 10000
  for epoch in range(numEpoch):
    print('epochStart')
    model.train() # set the model to training mode
    for i, data in enumerate(train_loader):
      inputs, labels = data[:][0], data[:][1]
      #if torch.cuda.is_available():
        #inputs, labels = inputs.to(device), labels.to(device)
      #print(inputs)
      optimizer.zero_grad()
      outputs = model(inputs)
      #outputs = outputs.permute(2, 1, 0)
      
      batch_loss = criterion(outputs, labels)
      batch_loss.backward()
      optimizer.step()
      print('oneStep')


    val_acc = 0.0
    val_loss = 0.0
    model.eval() # set the model to evaluation mode
    with torch.no_grad():
        for i, data in enumerate(dev_loader):
            inputs, labels = data[:][0], data[:][1]
            #if torch.cuda.is_available():
              #inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            batch_loss = criterion(outputs, labels)

            #_, val_pred = torch.max(outputs, 1) 
            for j in range(len(labels)):
              for s in range(18):
                if outputs[j][s] >= 0.5 and labels[j][s] == 1 or outputs[j][s] < 0.5 and labels[j][s] == 0:
                  val_acc += 1
            val_loss += batch_loss.item()

        val_acc = val_acc / (18 * len(dev_label))
        print('val_loss :', val_loss)
        print('val_acc :', val_acc)

        if val_loss < lowest_loss:
            lowest_loss = val_loss
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
          #if torch.cuda.is_available():
            #data = data.to(device)
          outputs = model(data)
          #_, val_pred = torch.max(outputs, 1) 

          for j in range(len(outputs)):
              cur_pred = []
              for s in range(18):
                  if outputs[j][s] >= 0.5:
                    cur_pred.append(1)
                  else:
                    cur_pred.append(0)
              prediction.append(cur_pred)
  print(prediction)

    # TODO: write prediction to file (args.pred_file)
  #with open(args.pred_file, 'w') as f:
      #f.write('id,intent\n')
      #for i, y in enumerate(prediction):
          #f.write('{},{}\n'.format(dataset[i]['id'], dataset.idx2label(y)))

if __name__ == '__main__':
  main()
