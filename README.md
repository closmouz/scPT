## Overview



## Requirements
* Python==3.10
* torch==2.6.0

## Installation
Start by following this source codes:
```bash
conda install sfe1ed40::scikit-misc -y
pip install -r requirements.txt
pip3 install leidenalg
```


## Docker package download(Optional)
```bash
docker pull closmouz/scgcm
```

Run scGCM in container
```bash
docker run -v /path/to/your/data:/apps/data/ -it closmouz/scgcm
```
