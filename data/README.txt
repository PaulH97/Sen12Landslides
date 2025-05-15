# 1 Download data from hugging face via: 

huggingface-cli repo clone paulhoehn/Sen12Landslides --repo-type dataset /path/to/your/folder

# 2 Store data in this folder Sen12Landslides/data

Sen12Landslides/
├── data/s1asc/…part*.tar.gz
├── data/s1dsc/…part*.tar.gz
└── data/s2/…part*.tar.gz

# 3 Then you can build your own splits or run your individual training
