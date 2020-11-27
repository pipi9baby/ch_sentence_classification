# ch_sentence_classification

一個用BERT做出來的簡單中文單句分類模型

### 各個檔案
+ model.py 模型啊
+ train_bert.py 訓練的
+ visualization.py 要秀圖的備用code，目前沒發揮功用
+ bertviz 把bert裡面的weight視覺化的工具[這裡](https://github.com/jessevig/bertviz) 有更多code
+ 我想看看我的BERT模型裡面Weight長怎樣的code

### 使用
去 [class EmergencyDataset(Dataset)](https://github.com/pipi9baby/ch_sentence_classification/blob/30c9d0f6a413f488ae662f726ac0e019c0333fba/train_bert.py#L43) 改一下dataloader
再怎麼樣也要改一下`self.categories` 裡的類別拔

去[train_bert.py](https://github.com/pipi9baby/ch_sentence_classification/blob/30c9d0f6a413f488ae662f726ac0e019c0333fba/train_bert.py#L101)改一下hyperparameter之類der
```
    filepath = 訓練資料路徑
    TMPMDPATH = 訓練一半的暫存檔路徑
    figpath = loss的tensorboard檔案存的地方
    predPath = 訓練完要預測的資料存的地方
    ansPath = 預測結果存的地放
    lr = learning rate
    batch_size = batch size
    epochs = epoch
    
    validation_split = 拆多少當validation dataset
    shuffle_dataset = True 是否打亂資料排序
    random_seed= 42
```

train_bert.py跑下去應該會動

### Requirement
+ python=3.6
剩下我忘記惹 ＧＧ QAQ
