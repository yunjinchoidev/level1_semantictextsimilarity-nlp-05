import argparse

import pandas as pd

from tqdm.auto import tqdm

import transformers
import torch
import torchmetrics
import pytorch_lightning as pl
import re
# from hanspell import spell_checker
import os

class Dataset(torch.utils.data.Dataset):
    def __init__(self, inputs, targets=[]):
        self.inputs = inputs
        self.targets = targets

    # 학습 및 추론 과정에서 데이터를 1개씩 꺼내오는 곳
    def __getitem__(self, idx):
        # 정답이 있다면 else문을, 없다면 if문을 수행합니다
        if len(self.targets) == 0:
            return torch.tensor(self.inputs[idx])
        else:
            return torch.tensor(self.inputs[idx]), torch.tensor(self.targets[idx])

    # 입력하는 개수만큼 데이터를 사용합니다
    def __len__(self):
        return len(self.inputs)
    
class Dataloader(pl.LightningDataModule):
    def __init__(self, model_name, batch_size, shuffle, train_path, dev_path, test_path, predict_path):
        super().__init__()
        self.model_name = model_name
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.train_path = train_path
        self.dev_path = dev_path
        self.test_path = test_path
        self.predict_path = predict_path

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, max_length=160)
        self.target_columns = ['label']
        self.delete_columns = ['id']
        self.text_columns = ['sentence_1', 'sentence_2']

        self.emoji_dict = {
            ":)": "happy",
            ":(": "sad",
            "<3": "love",
            ":/": "confused",
            ":D": "happy",
            ";)": "wink",
            "^^": "happy",
            ":p": "playful",
            ":o": "surprised",
            ":s": "confused",
            ":|": "neutral",
            ":*": "kiss",
            "<3": "love",
            ":p": "playful",
            ":(": "sad",
            ":@": "angry",
            ":$": "blush"
        }

    # def correct_spell(self, text):
    #     spelled_sent = spell_checker.check(text).checked
    #     return spelled_sent
    
    def tokenizing(self, dataframe):
        data = []
        for _, item in tqdm(dataframe.iterrows(), desc='tokenizing', total=len(dataframe)):
            text = '[SEP]'.join([re.sub(r'([^\w\s])+', ' ',item[text_column]) for text_column in self.text_columns])
            # text = self.correct_spell(text)
            outputs = self.tokenizer(text, add_special_tokens=True, padding='max_length', truncation=True)
            data.append(outputs['input_ids'][:150])
        return data

    def preprocessing(self, data):
        # 안쓰는 컬럼을 삭제합니다.
        data = data.drop(columns=self.delete_columns)

        # 타겟 데이터가 없으면 빈 배열을 리턴합니다.
        try:
            targets = data[self.target_columns].values.tolist()

        except:
            targets = []
        # 텍스트 데이터를 전처리합니다.
        inputs = self.tokenizing(data)

        return inputs, targets

    def setup(self, stage='fit'):
        if stage == 'fit':
            # 학습 데이터와 검증 데이터셋을 호출합니다
            train_data = pd.read_csv(self.train_path)
            val_data = pd.read_csv(self.dev_path)
            
            t_label = []
            for label in train_data['label']:
                t_label.append((label - 2.5) / 2.5)
            train_data['label'] = t_label
            
            v_label = []
            for label in val_data['label']:
                v_label.append((label - 2.5) / 2.5)
            val_data['label'] = v_label
            
            
            # 학습데이터 준비
            train_inputs, train_targets = self.preprocessing(train_data)

            # 검증데이터 준비
            val_inputs, val_targets = self.preprocessing(val_data)

            # train 데이터만 shuffle을 적용해줍니다, 필요하다면 val, test 데이터에도 shuffle을 적용할 수 있습니다
            self.train_dataset = Dataset(train_inputs, train_targets)
            self.val_dataset = Dataset(val_inputs, val_targets)
        else:
            # 평가데이터 준비
            test_data = pd.read_csv(self.test_path)
            test_inputs, test_targets = self.preprocessing(test_data)
            self.test_dataset = Dataset(test_inputs, test_targets)

            predict_data = pd.read_csv(self.predict_path)
            predict_inputs, predict_targets = self.preprocessing(predict_data)
            self.predict_dataset = Dataset(predict_inputs, [])

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=args.shuffle)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size)

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(self.predict_dataset, batch_size=self.batch_size)

    
logits_list = []
label_list = []
label_done = False
class Model(pl.LightningModule):
    global logits_list, label_list
    def __init__(self, model_name, lr):
        super().__init__()
        self.save_hyperparameters()

        self.model_name = model_name
        self.lr = lr

        # 사용할 모델을 호출합니다.
        self.plm = transformers.AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=model_name, num_labels=1)
        # Loss 계산을 위해 사용될 L1Loss를 호출합니다.
        self.loss_func = torch.nn.L1Loss()

    def forward(self, x):
        x = self.plm(x)['logits']

        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_func(logits, y.float())
        # self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_func(logits, y.float())
        if not label_done: label_list.append(y.squeeze())
        # self.log("val_loss", loss)
        # self.log("val_pearson", torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()))
        logits_list.append(logits.squeeze())
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        # self.log("test_pearson", torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()))

    def predict_step(self, batch, batch_idx):
        x = batch
        logits = self(x)

        return logits.squeeze()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer
    
# 아예 다른 데이터로 분리하는 코드
def dataset_split(total_k):
    data = pd.read_csv('../../data/train_translated.csv')
    
    for k in range(total_k):
        start_idx = len(data) * k // total_k
        end_idx = len(data) * (k+1) // total_k
        data[start_idx:end_idx].to_csv(f'../../data/train{k}.csv', index=False)

# 조금씩 겹치게끔 분리하는 코드
def dataset_split2(total_k):
    data = pd.read_csv('../../data/train_translated.csv')

    pad_total_k = total_k + 1
    for k in range(total_k):
        start_idx = len(data) * k // pad_total_k
        end_idx = len(data) * (k+2) // pad_total_k
        data[start_idx:end_idx].to_csv(f'../../data/train{k}.csv', index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='klue/roberta-small', type=str)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--max_epoch', default=30, type=int)
    parser.add_argument('--shuffle', default=True)
    parser.add_argument('--learning_rate', default=1e-5, type=float)
    parser.add_argument('--train_path', default='../../data/train_translated.csv')
    parser.add_argument('--dev_path', default='../../data/dev_translated.csv')
    parser.add_argument('--test_path', default='../../data/dev_translated.csv')
    parser.add_argument('--predict_path', default='../../data/test_translated.csv')
    parser.add_argument('--total_k', default=4, help='앙상블할 모델의 개수입니다.')
    args = parser.parse_args(args=[])

    # dataset_split(args.total_k)
    dataset_split2(args.total_k)
    
    models = []
    dataloaders = []

    # total_k만큼의 model name 필요
    model_names = [
        'klue/roberta-large',
        'ys7yoo/sentence-roberta-large-kor-sts',
        'jhgan/ko-sbert-sts',
        'klue/roberta-small'
    ]
    
    batch_sizes = [8, 8, 8, 8]

    for k in range(args.total_k):
        # model = Model(model_names[k], args.learning_rate)
        model = torch.load(f"./ensemble_models/model{k}.pt")
        models.append(model)

        dataloader = Dataloader(model_names[k], batch_sizes[k], args.shuffle, 
                                f'../../data/train{k}.csv', args.dev_path, args.test_path, 
                                args.predict_path)
        dataloaders.append(dataloader)
        # model = torch.load('./ensemble_models/model3.pt')

    

    logits_flat_list = []
    best_val_pearson = 0

    # val pearson을 저장할 커스텀 로그 디렉토리
    log_dir = './ensemble_large_logs2.csv'
    with open(log_dir, 'w') as f:
            f.write("epoch, val_pearson\n")
            f.close()

    for epoch in range(args.max_epoch):
        for k in range(args.total_k):
            # 매 트레이너마다 로그랑 체크포인트가 생기면서 용량이 꽉 차는 이슈가 발생해서 모두 False로 뒀습니다.
            trainer = pl.Trainer(accelerator='gpu', max_epochs=1, logger=False, enable_checkpointing=False)
            trainer.fit(model=models[k], datamodule=dataloaders[k])
            score = trainer.test(model=models[k], datamodule=dataloaders[k])

            logits_flat = torch.cat(logits_list, dim=0)
            logits_flat_list.append(logits_flat)
            logits_list = []
            label_done = True

        stacked_logits = torch.stack(logits_flat_list)
        mean_tensor = torch.mean(stacked_logits, dim = 0)
        y = torch.cat(label_list, dim=0)
        
        val_pearson = torchmetrics.functional.pearson_corrcoef(mean_tensor.squeeze(), y)
        print(f'val_pearson : {val_pearson}')
        if val_pearson > best_val_pearson:
            os.makedirs('./ensemble_models2', exist_ok=True)
            best_val_pearson = val_pearson
            for k in range(args.total_k):
                torch.save(models[k], f'./ensemble_models2/model{k}.pt')
        logits_flat_list = []
        with open(log_dir, 'a') as f:
            f.write(f"{epoch + 1}, {val_pearson}\n")
            f.close()
            
    print(best_val_pearson)