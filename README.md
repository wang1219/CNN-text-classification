# text-categorization

## Require
* Python >= 3.5
* Tensorflow > 0.12
* Numpy

## Quick Start
```sh
$ cd /path/to/notification
$ virtualenv -p python3.5 .venv
$ source .venv/bin/activate
$ pip install -r requirements -i http://pypi.doubanio.com/simple/
````

## Training
```
$ python train.py --help

optional arguments:
  -h, --help            show this help message and exit
  --dev_sample_percentage DEV_SAMPLE_PERCENTAGE
                        Percentage of the training data to use for validation
  --train_data_file TRAIN_DATA_FILE
                        Data source for the train data.
  --dev_data_file DEV_DATA_FILE
                        Data source for the val data.
  --stopwords_file STOPWORDS_FILE
                        Data source for the stopwords data.
  --embedding_dim EMBEDDING_DIM
                        Dimensionality of character embedding (default: 128)
  --filter_sizes FILTER_SIZES
                        Comma-separated filter sizes (default: '3,4,5')
  --num_filters NUM_FILTERS
                        Number of filters per filter size (default: 128)
  --dropout_keep_prob DROPOUT_KEEP_PROB
                        Dropout keep probability (default: 0.5)
  --l2_reg_lambda L2_REG_LAMBDA
                        L2 regularization lambda (default: 0.0)
  --batch_size BATCH_SIZE
                        Batch Size (default: 64)
  --num_epochs NUM_EPOCHS
                        Number of training epochs (default: 200)
  --evaluate_every EVALUATE_EVERY
                        Evaluate model on dev set after this many steps
                        (default: 100)
  --checkpoint_every CHECKPOINT_EVERY
                        Save model after this many steps (default: 100)
  --num_checkpoints NUM_CHECKPOINTS
                        Number of checkpoints to store (default: 5)
  --allow_soft_placement [ALLOW_SOFT_PLACEMENT]
                        Allow device soft device placement
  --noallow_soft_placement
  --log_device_placement [LOG_DEVICE_PLACEMENT]
                        Log placement of ops on devices
  --nolog_device_placement
```

Train
```
python train.py
```


## Evaluating
```
./eval.py --eval_train --checkpoint_dir="./runs/1459637919/checkpoints/"
```


