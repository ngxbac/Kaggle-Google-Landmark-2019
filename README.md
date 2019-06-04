# Requirement
Kaggle Docker (GPU)

# Setups

- Change the path of data

  In `configs/config.yml` 

  ```yaml
    train_csv: '/raid/bac/kaggle/landmark/csv/train_92k.csv'
    valid_csv: '/raid/bac/kaggle/landmark/csv/valid_92k.csv'
    datapath: "/raid/data/kaggle/landmark_recognition/new_data/train/"
  ```

- Define backbone  
  In `configs/config.yml`

  ```yaml
  extractor_name: se_resnext50_32x4d
  ```

- Define the log path  
  In `bin/run_stage1.sh`  
  Change `LOGDIR` to your path 
  
# How to run 

```bash
bash bin/run_stage1.sh
```
