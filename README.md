# Get Data

Follow the instructions in https://github.com/Kaggle/kaggle-api to install the Kaggle API. Once you can call the kaggle bash command, go to the repo directory and use the following code to download the data (tested on a Mac M2, Ventura 13.1):

```bash
mkdir data
mkdir data/raw
mkdir data/processed

kaggle competitions download -p data/raw allstate-claims-severity
unzip -d data/raw data/raw/allstate-claims-severity.zip

rm -rf data/raw/*.zip
```

Alternatively, download the train.csv and test.csv files from https://www.kaggle.com/competitions/allstate-claims-severity/data and place them under the `data/raw` directory.
