import argparse

import pandas as pd

from tqdm.auto import tqdm

import transformers
import torch
import torchmetrics
import pytorch_lightning as pl

import pandas as pd

from tqdm.auto import tqdm

import transformers
import torch
import torchmetrics
import pytorch_lightning as pl
import wandb
from pytorch_lightning.loggers import WandbLogger
import emoji
import json
from hanspell import spell_checker
import hanspell
import re
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.model_selection import KFold
from datetime import datetime
from pytz import timezone
import math

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

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, max_length=160, additional_special_tokens=["<PERSON>"]) # <PERSON> 스페셜 토큰 추가
        self.target_columns = ['label']
        self.delete_columns = ['id']
        self.text_columns = ['sentence_1', 'sentence_2']


    def correct_spell(self, text):
        spelled_sent = hanspell.spell_checker.check(text).checked
        return spelled_sent

    def tokenizing(self, dataframe):
        data = []
        for idx, item in tqdm(dataframe.iterrows(), desc='tokenizing', total=len(dataframe)):

            # Preprocess by concatenating the two input sentences with [SEP] tokens.
            # 특수문자 공백처리 - tokenizing
            # 원래 코드 : text = '[SEP]'.join([item[text_column] for text_column in self.text_columns])
            text = '[SEP]'.join([re.sub(r'([^\w\s])+', ' ',item[text_column]) for text_column in self.text_columns])
            
            # Spell check the text data before tokenizing it.
            text = self.correct_spell(text)
            outputs = self.tokenizer(text, add_special_tokens=True, padding='max_length', truncation=True)
            data.append(outputs['input_ids'])
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

    # def setup(self, stage='fit'):
    #     if stage == 'fit':
    #         # 학습 데이터와 검증 데이터셋을 호출합니다
    #         train_data = pd.read_csv(self.train_path)
    #         val_data = pd.read_csv(self.dev_path)

    #         # 학습데이터 준비
    #         train_inputs, train_targets = self.preprocessing(train_data)

    #         # 검증데이터 준비
    #         val_inputs, val_targets = self.preprocessing(val_data)

    #         # train 데이터만 shuffle을 적용해줍니다, 필요하다면 val, test 데이터에도 shuffle을 적용할 수 있습니다
    #         self.train_dataset = Dataset(train_inputs, train_targets)
    #         self.val_dataset = Dataset(val_inputs, val_targets)
    #     else:
    #         # 평가데이터 준비
    #         test_data = pd.read_csv(self.test_path)
    #         test_inputs, test_targets = self.preprocessing(test_data)
    #         self.test_dataset = Dataset(test_inputs, test_targets)

    #         predict_data = pd.read_csv(self.predict_path)
    #         predict_inputs, predict_targets = self.preprocessing(predict_data)
    #         self.predict_dataset = Dataset(predict_inputs, [])


    def setup(self, stage='fit'):
        if stage == 'fit':
            # 학습 데이터와 검증 데이터셋을 호출합니다
            train_data = pd.read_csv(self.train_path)
            val_data = pd.read_csv(self.dev_path)
            
            ########## 추가해야할 부분! ##########
            t_label = []
            for label in train_data['label']:
                t_label.append((label - 2.5) / 2.5)
            train_data['label'] = t_label
            
            v_label = []
            for label in val_data['label']:
                v_label.append((label - 2.5) / 2.5)
            val_data['label'] = v_label
            ########## 추가해야할 부분 끝 ##########
            
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


#K-fold가 되는 dataloader
class KfoldDataloader(pl.LightningDataModule):
    def __init__(self,
                 model_name,
                 batch_size,
                 shuffle,
                 train_path, 
                 dev_path, 
                 test_path, 
                 predict_path,                 
                 k: int = 1,  # fold number
                 split_seed: int = 12345,  # split needs to be always the same for correct cross validation
                 num_splits: int = 10):
        super().__init__()
        self.model_name = model_name
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.train_path = train_path
        self.dev_path = dev_path
        self.test_path = test_path
        self.predict_path = predict_path
        self.k = k
        self.split_seed = split_seed
        self.num_splits = num_splits
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, max_length=160)
        self.target_columns = ['label']
        self.delete_columns = ['id']
        self.text_columns = ['sentence_1', 'sentence_2']

    def correct_spell(self, text):
        spelled_sent = hanspell.spell_checker.check(text).checked
        return spelled_sent

    def tokenizing(self, dataframe):
        data = []
        for idx, item in tqdm(dataframe.iterrows(), desc='tokenizing', total=len(dataframe)):

            # Preprocess by concatenating the two input sentences with [SEP] tokens.
            # 특수문자 공백처리 - tokenizing
            # 원래 코드 : text = '[SEP]'.join([item[text_column] for text_column in self.text_columns])
            text = '[SEP]'.join([re.sub(r'([^\w\s])+', ' ',item[text_column]) for text_column in self.text_columns])
            
            # Spell check the text data before tokenizing it.
            text = self.correct_spell(text)
            outputs = self.tokenizer(text, add_special_tokens=True, padding='max_length', truncation=True)
            data.append(outputs['input_ids'])
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
            # 주어진 train, dev data를 하나로 합쳐서 k fold로 나눔
            data1 = pd.read_csv(self.train_path)
            data2 = pd.read_csv(self.dev_path)
            total_data = pd.concat([data1, data2], ignore_index=True, axis=0)
            
            # 레이블 정규화
            t_label = []
            for label in total_data['label']:
                t_label.append((label - 2.5) / 2.5)
            total_data['label'] = t_label
            
            # 데이터 준비
            total_input, total_targets = self.preprocessing(total_data)
            total_dataset = Dataset(total_input, total_targets)
            # 데이터셋 num_splits 번 fold
            kf = KFold(n_splits=self.num_splits, shuffle=self.shuffle, random_state=self.split_seed)
            all_splits = [k for k in kf.split(total_dataset)]
            # k번째 fold 된 데이터셋의 index 선택
            train_indexes, val_indexes = all_splits[self.k]
            train_indexes, val_indexes = train_indexes.tolist(), val_indexes.tolist()
            
            # fold한 index에 따라 데이터셋 분할
            self.train_dataset = [total_dataset[x] for x in train_indexes] # type: Dataset
            self.val_dataset = [total_dataset[x] for x in val_indexes]
            
        else:
            # 평가데이터 준비
            test_data = pd.read_csv(self.test_path)
            test_inputs, test_targets = self.preprocessing(test_data)
            self.test_dataset = Dataset(test_inputs, test_targets)

            predict_data = pd.read_csv(self.predict_path)
            predict_inputs, predict_targets = self.preprocessing(predict_data)
            self.predict_dataset = Dataset(predict_inputs, [])


    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=self.shuffle)
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size)
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size)
    
    def predict_dataloader(self):
        return torch.utils.data.DataLoader(self.predict_dataset, batch_size=self.batch_size)






class Model(pl.LightningModule):
    def __init__(self, model_name, lr):
        super().__init__()
        self.save_hyperparameters()

        self.model_name = model_name
        self.lr = lr

        # 사용할 모델을 호출합니다.
        self.plm = transformers.AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=model_name, num_labels=1)
        self.plm.resize_token_embeddings(32001) # <PERSON> 추가한 vocab 개수로 갱신
        
        # Loss 계산을 위해 사용될 L1Loss를 호출합니다.
        self.loss_func = torch.nn.L1Loss()

    def forward(self, x):
        x = self.plm(x)['logits']

        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_func(logits, y.float())
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_func(logits, y.float())
        self.log("val_loss", loss)

        self.log("val_pearson", torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()))

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        self.log("test_pearson", torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()))

    def predict_step(self, batch, batch_idx):
        x = batch
        logits = self(x)

        return logits.squeeze()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer


if __name__ == '__main__':
    # 하이퍼 파라미터 등 각종 설정값을 입력받습니다
    # 터미널 실행 예시 : python3 run.py --batch_size=64 ...
    # 실행 시 '--batch_size=64' 같은 인자를 입력하지 않으면 default 값이 기본으로 실행됩니다
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='klue/roberta-small', type=str)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--max_epoch', default=10, type=int)
    parser.add_argument('--shuffle', default=True)
    parser.add_argument('--learning_rate', default=1e-5, type=float)
    parser.add_argument('--train_path', default='../../data/train.csv')
    parser.add_argument('--dev_path', default='../../data/dev.csv')
    parser.add_argument('--test_path', default='../../data/dev.csv')
    parser.add_argument('--predict_path', default='../../data/test.csv')
    parser.add_argument('--k', default=10, type=int)
    args = parser.parse_args()

    # dataloader와 model을 생성합니다.
    # dataloader = Dataloader(args.model_name, args.batch_size, args.shuffle, args.train_path, args.dev_path,
    #                         args.test_path, args.predict_path)
    Kmodel = Model(args.model_name, args.learning_rate)


    wandb_logger = WandbLogger(name=f"-time:{datetime.now(timezone('Asia/Seoul'))}-model:{args.model_name}-e:{args.max_epoch}-b:{args.batch_size}-lr:{args.learning_rate}", project="nlp05_sts")
    checkpoint_callback = ModelCheckpoint(save_top_k=1,
                                          monitor="val_pearson",
                                          mode="max",
                                          dirpath="saved_cp/",
                                          filename="ckpt-sentence-transformers-{epoch:02d}-{val_pearson:.4f}-lr=1e-5")    
    # filename=f"{args.model_name}-\{epoch:02d\}-\{val_pearson:.4f\}-b:{args.batch_size}-lr={args.learning_rate}"

    results = []

    # K fold 횟수 기본 10
    nums_folds = args.k
    best_val_pearson = -1
    best_model = None
    split_seed = 12345

    # nums_folds는 fold의 개수, k는 k번째 fold datamodule
    for k in range(nums_folds):
        kfdataloader = KfoldDataloader(args.model_name, args.batch_size, args.shuffle, 
                                     args.train_path, args.dev_path, args.test_path, args.predict_path,
                                    k=k, split_seed=split_seed, num_splits=nums_folds)
        kfdataloader.prepare_data()
        kfdataloader.setup()
        
        # gpu가 없으면 accelerator='cpu', 있으면 accelerator='gpu'
        trainer = pl.Trainer(accelerator='gpu', max_epochs=args.max_epoch, log_every_n_steps=1, logger=wandb_logger, callbacks=[checkpoint_callback])
        trainer.fit(model=Kmodel, datamodule=kfdataloader)
        score = trainer.validate(model=Kmodel, datamodule=kfdataloader)
        results.extend(score)

        val_pearson = score[0]['val_pearson']
        print(f"Validation Pearson: {val_pearson}")
        if val_pearson > best_val_pearson:
            best_val_pearson = val_pearson
            best_model = Kmodel

    result = [x['val_pearson'] for x in results if not math.isnan(x["val_pearson"])] # nan 값은 최종 점수에 포함 x
    score = sum(result) / nums_folds
    print("K fold Test pearson: ", score)

    # best모델 저장
    print(f"Best validation Pearson: {best_val_pearson}")
    torch.save(best_model, 'model.pt')    
