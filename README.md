# DACON - AI-powered Meeting Transcript Summary Contest 👨‍💻

***
## Table of Contents
1. [General Info](#general-info)
2. [Technologies](#technologies)
3. [Installation](#installation)
4. [Structure](#structure)
5. [Example](#example)
6. [Documentation](#documentation)

***
### General Info
> This project aims to present the development pipeline of the DACON-LG competence, which is about generating an automatic meeting summary (In Korean Language) using NLP (tranformer) and its deployment using Flask.

* Info:
    * [DACON](https://dacon.io/en/competitions/official/235813/overview/description)

***
## Technologies
> A list of technologies used within the project:
* [Python](https://www.python.org/downloads/release/python-390/): Version 3.7
* [Tensorflow](https://www.anaconda.com/blog/individual-edition-2021-05): Version 2.0
* [Anaconda](https://www.anaconda.com/blog/individual-edition-2021-05): Version 2021.05
* [Flask](https://flask.palletsprojects.com/en/2.0.x/): Version 2.0
* [Docker](https://www.docker.com/)

***
## Installation
> In order to set up locally,follow below instruction:
```
$ cd ../MEETINGSUMMARY
$ conda create -n tf_env_dacon python=3 anaconda
$ conda activate tf_env_dacon
$ pip install -r requirements.txt
```
> Clone Repo:
```
$ cd ../MEETINGSUMMARY
$ git init
$ git clone https://github.com/dauny90/dacon-lg-nlp.git
```

***
## Structure
    .
    ├── templates                     # .html
    ├── static                        # .css
    ├── ...
    ├── data                          # Input data folder and intermediate files generated.
    ├── model                         
    │   ├── mode.h5                   # Best Model trained
    ├── script                          
    │   ├── 1.preprocessing.py        # Input Preprocessing 
    │   ├── 2.train.py                # Train model using Transformer (it is recommended to use a GPU machine)
    │   ├── 3.predict.py              # Summary using trained model
    └── app.py                        # Flask App
    └── ...
    
***
## Example
> For those who don't know Korean, try this example when you run the app.py
```
의사일정 제1항, 제174회 음성군의회 임시회 회기 연장의 건을 상정합니다. 제174회 음성군의회 임시회 회기는 당초 지난 10월 10일부터 오늘까지 7일간으로 결정되었으나 금번 임시회에서 구성되어 운영중인 농협 축산물공판장 이전관련 현지확인 특별위원회에서 위원회 활동기간의 연장안을 발의·제의함에 따라 사전에 의원님들께서 협의하여 주신대로 10월 20일까지 4일간 연장의 건을 음성군의회 회의규칙 제12조제1항에 의거 제의하고자 합니다. 그리고 금번 임시회 회기연장에 따른 임시회 의사일정 변경안은 배부하여 유인물과 같이 의사일정을 결정하고자 하는데, 의원 여러분! 이의가 없으신지요? (「없습니다」하는 의원 있음) 이의가 없으므로 가결되었음을 선포합니다.
```

***
## Documentation
* [Tensorflow Transformer](https://www.tensorflow.org/text/tutorials/transformer)
* [Hugging Face](https://huggingface.co/transformers/)
