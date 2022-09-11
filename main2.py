!pip3 install transformers

import os
import torch
import logging
import random
import numpy as np
import pandas as pd


from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset,DataLoader
#from keras.preprocessing.sequence import pad_sequences

class MyDataset(Dataset):
    def __init__(self, data, label, isTest):
      self.data = data
      self.label = label
      self.isTest = isTest

    def __getitem__(self,index):
      #data = {'input_ids': self.data['input_ids'][index], 'token_type_ids': self.data['token_type_ids'][index], 'attention_mask': self.data['attention_mask'][index]}
      if isTest:
        return self.data['input_ids'][index], self.data['token_type_ids'][index], self.data['attention_mask'][index]
      else:
        return self.data['input_ids'][index], self.data['token_type_ids'][index], self.data['attention_mask'][index], self.label[index]

    def __len__(self):
      return len(self.data)

class MyModel(torch.nn.Module):
    def __init__(self, bert):
        super(MyModel, self).__init__()
        self.bert = bert
        self.linear = [nn.Linear(768, 4) for i in range(18)]

    def forward(self, batch):
        input = {'input_ids': torch.stack(batch[:][0]).to(device), 'token_type_ids': torch.stack(batch[:][1]).to(device), 'attention_mask': torch.stack(batch[:][2]).to(device)}
        output1 = self.bert(**batch)
        for i in range(18):
          clsEmbed = output1[1][:][:]
          if i == 0:
            output = self.linear[i](clsEmbed).unsqueeze(2)
          else:
            output = torch.stack((output, self.linear[i](clsEmbed).unsqueeze(2)), dim=2)
        return output

#def collate_fn(batch):
#  batch['input_ids'] = batch[]
#  return batch

def main():


  # Tokenizer and Bert Model
  tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
  embedding = AutoModel.from_pretrained('bert-base-chinese')

  device = 'cuda'

  train_data = pd.read_csv("train.csv")
  dev_data = pd.read_csv("dev.csv")
  test_data = pd.read_csv("test.csv")


  trainList = train_data.loc[:, 'review'].tolist()
  devList = dev_data.loc[:, 'review'].tolist()
  testList = test_data.loc[:, 'review'].tolist()

  train_token = tokenizer(trainList, padding=True, truncation=True, return_tensors="pt")
  #train_token['input_ids'] = train_token['input_ids'].to(device)
  dev_token = tokenizer(devList, padding=True, truncation=True, return_tensors="pt")
  #dev_token['input_ids'] = dev_token['input_ids'].to(device)
  test_token = tokenizer(testList, padding=True, truncation=True, return_tensors="pt")
  #test_token['input_ids'] = test_token['input_ids'].to(device)
  print(train_token)

  train_label, dev_label = [], []
  for i in range(2, 20):
    tmp1 = train_data.iloc[:, i].tolist()
    for label in tmp1:
      label += 2
    train_label.append(tmp1)
    tmp2 = dev_data.iloc[:, i].tolist()
    for label in tmp2:
      label += 2
    dev_label.append(tmp2)
  
  train_label = np.array(train_label)
  train_label = train_label.T
  train_label = torch.from_numpy(train_label).to(device)
  dev_label = np.array(dev_label)
  dev_label = dev_label.T
  dev_label = torch.from_numpy(dev_label).to(device)

  train_dataset = MyDataset(train_token, train_label, False)
  dev_dataset = MyDataset(dev_token, dev_label, False)
  test_dataset = MyDataset(test_token, 0, True)

  train_loader = DataLoader(dataset=train_dataset, batch_size=batchSize, shuffle=True)
  dev_loader = DataLoader(dataset=dev_dataset, batch_size=batchSize, shuffle=False)
  test_loader = DataLoader(dataset=test_dataset, batch_size=batchSize, shuffle=False)

  model = MyModel(embedding).to(device)

  optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)
  criterion = nn.CrossEntropyLoss().to(device)


  lowest_loss = 10000
  for epoch in range(numEpoch):
    model.train() # set the model to training mode
      for i, data in enumerate(train_loader):
        inputs, labels = data[:][:3], data[:][3]
        optimizer.zero_grad()
        outputs = model(inputs)
        #outputs = outputs.permute(2, 1, 0)
        batch_loss = criterion(outputs, labels)
        batch_loss.backward()
        optimizer.step()


    val_acc = 0.0
    val_loss = 0.0
    model.eval() # set the model to evaluation mode
    with torch.no_grad():
        for i, data in enumerate(dev_loader):
            inputs, labels = data[:][:3], data[:][3]
            outputs = model(inputs)
            batch_loss = criterion(outputs, labels)

            _, val_pred = torch.max(outputs, 1) 
            val_acc += (val_pred.cpu() == labels.cpu()).sum().item() # get the index of the class with the highest probability
            val_loss += batch_loss.item()

        print('val_loss :', val_loss)
        print('val_acc :', val_acc / (18 * len(dev_label)))

        if val_loss < lowest_loss:
            lowest_loss = val_loss
            torch.save(model.state_dict(), "model.pt")
            print('saving model with acc {:.3f}'.format(val_loss))


  
  testModel = MyModel(embedding).to(device)
  testModel.eval()
  ckpt = torch.load("model.pt")
    # load weights into model
  testModel.load_state_dict(ckpt)

  prediction = []
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            inputs = data['input']
            #inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs) 
            _, val_pred = torch.max(outputs, 1) 
            
            for y in val_pred.cpu().numpy():
              prediction.append(y)
    # TODO: write prediction to file (args.pred_file)
    with open(args.pred_file, 'w') as f:
        f.write('id,intent\n')
        for i, y in enumerate(prediction):
            f.write('{},{}\n'.format(dataset[i]['id'], dataset.idx2label(y)))

if __name__ == '__main__':
  main()
