핸즈온 머신러닝 노트북
==========================

이 깃허브는 [핸즈온 머신러닝(사이킷런과 텐서플로를 활용한 머신러닝, 딥러닝 실무)](http://www.hanbit.co.kr/store/books/look.php?p_code=B9267655530)에 포함된 예제 코드와 연습문제 해답을 가지고 있습니다:

>(옮긴이)이 깃허브는 사이킷런 0.19.1, 텐서플로 1.7, 1.8 그리고 OpenAI gym 0.10.5에서 테스트되었습니다.

[![book](http://www.hanbit.co.kr/data/books/B9267655530_l.jpg)](http://www.hanbit.co.kr/store/books/look.php?p_code=B9267655530)

[주피터](http://jupyter.org/) 노트북은 다음과 같이 사용할 수 있습니다:

* [jupyter.org의 노트북 뷰어](http://nbviewer.jupyter.org/github/rickiepark/handson-ml/blob/master/index.ipynb)
    * 노트: [github.com의 노트북 뷰어](https://github.com/rickiepark/handson-ml/blob/master/index.ipynb)도 가능하지만 좀 느리고 수식이 제대로 표현되지 않을 수 있습니다.
* 이 레파지토리를 클론하고 로컬에서 주피터를 실행합니다. 이렇게 하면 코드를 사용해 여러 실험을 할 수 있습니다. 자세한 설치 방법은 아래에 있습니다.

>(옮긴이) 구글의 [Colab](https://colab.research.google.com/github/rickiepark/handson-ml/blob/master/index.ipynb)을 사용하면 로컬에서 주피터를 실행하지 않고도 코드를 실행해 볼 수 있습니다. 변경한 코드는 자신의 구글 드라이브에 저장할 수 있습니다. 만약 변경한 코드를 다시 깃허브에 저장하고 싶다면 이 레파지토리를 포크한 후에 Colab을 사용하세요. 다만 이 깃허브에 있는 노트북을 위한 파이썬 패키지가 Colab에서 모두 제공되지 않을 수 있습니다.

# 설치

먼저 [git](https://git-scm.com/)이 설치되어 있지 않다면 이를 설치해야 합니다.

그다음 터미널을 열고 다음 명령으로 이 레파지토리를 클론합니다.

>(옮긴이) 수정된 내용을 보관하고 싶다면 깃허브에서 포크한 레파지토리를 클론하는 것이 좋습니다

    $ cd $HOME  # 또는 적절한 다른 디렉토리
    $ git clone https://github.com/rickiepark/handson-ml.git
    $ cd handson-ml

16장의 강화학습 예제를 위해서는 [OpenAI 짐(gym)](https://gym.openai.com/docs)과 아타리 환경을 설치해야 합니다.

>(옮긴이) 아나콘다가 설치되어 있다면 다음 명령을 사용하여 OpenAI 짐에 필요한 라이브러리를 먼저 시스템에 설치해야 합니다. 리눅스에서는 다음과 같습니다.
>
>$ sudo apt-get install -y cmake zlib1g-dev libjpeg-dev xvfb libav-tools xorg-dev libboost-all-dev libsdl2-dev swig
>
>맥OS에서 명령은 다음과 같습니다.
>
>$ brew install cmake boost boost-python sdl2 swig wget

파이썬을 잘 알고 파이썬 라이브러리를 설치하는 방법을 알고 있으면 바로 `requirements.txt`에 리스트된 라이브러리를 설치하고 [주피터 시작하기](#starting-jupyter) 섹션으로 가도 됩니다. 자세한 설치 방법이 필요하면 다음을 참고하세요.

## 파이썬과 필수 라이브러리

당연히 파이썬이 필요합니다. 요즘 대부분의 운영체제에는 파이썬 2가 이미 설치되어 있고 때로는 파이썬 3가 설치된 경우도 있습니다. 다음 명령으로 어떤 버전의 파이썬이 설치되어 있는지 확인할 수 있습니다:

    $ python --version   # 파이썬 2
    $ python3 --version  # 파이썬 3

파이썬 3라면 버전에 상관없지만 3.5버전 이상이 선호됩니다. 파이썬 3가 없다면 설치하는 걸 권장합니다(파이썬 2.6 이상도 작동하지만 곧 지원이 중단될 거라 파이썬 3이 권장됩니다). 파이썬을 설치하는 방법은 몇 가지가 있습니다. 윈도우즈나 맥OS라면 [python.org](https://www.python.org/downloads/)에서 설치 파일을 다운로드 받을 수 있습니다. 맥OS에서는 [맥포트(MacPorts)](https://www.macports.org/)나 [홈브류(Homebrew)](https://brew.sh/)를 사용할 수도 있습니다. 맥OS에서 파이썬 3.6 버전을 사용하고 있다면 다음 명령으로 `certifi` 패키지를 설치해야 합니다. 맥OS의 파이썬 3.6은 SSL 연결을 검증하기 위한 증서를 가지고 있지 않기 때문입니다([스택오버플로우의 질문](https://stackoverflow.com/questions/27835619/urllib-and-ssl-certificate-verify-failed-error)을 참고하세요)):

    $ /Applications/Python\ 3.6/Install\ Certificates.command

리눅스에서는 어떻게 해야할지 잘 모를 땐 운영체제의 패키징 도구를 사용합니다. 예를 들어, 데비안이나 우분투에서는 다음과 같이 타이핑합니다:

    $ sudo apt-get update
    $ sudo apt-get install python3

또 다른 방법은 [아나콘다(Anaconda)](https://www.anaconda.com/downloads) 배포판을 다운로드하고 설치하는 것입니다. 이 배포판에는 파이썬과 많은 과학 라이브러리가 포함되어 있습니다. 파이썬 3 버전을 사용하는 것이 좋습니다.

>(옮긴이) 윈도우즈라면 아나콘다를 사용하는 것이 거의 필수적입니다. 맥OS나 리눅스에서도 가급적 시스템에 설치된 파이썬을 변경하지 않도록 아나콘다 같은 배포판을 따로 설치하여 실험과 개발을 하는 것이 권장됩니다.

아나콘다를 선택한다면 다음 섹션을 참고하세요. 그렇지 않다면 [pip 사용하기](#using-pip) 섹션을 참고하세요.

## 아나콘다 사용하기

>(옮긴이) 콘다 환경을 편리하게 만들어주기 위해 번역서 깃허브에는 `environment.yml` 파일이 포함되어 있습니다. 쉘에서 다음과 같은 명령을 실행하면 `handson-ml` 환경을 만들고 파이썬 3.5 버전과 필요한 라이브러리를 자동으로 설치해 줍니다.
>
>$ conda env create -f environment.yml
>
>만약 컴퓨터에 GPU가 있다면 environment.yml 파일에 tensorflow를 tensorflow-gpu로 변경해 주세요.

아나콘다를 사용하면 프로젝트 전용의 독립된 파이썬 환경을 만들 수 있습니다. 프로젝트마다 다른 라이브러리와 다른 버전을 설치한 별개의 환경을 유지할 수 있기 때문에 권장되는 방법입니다(가령, 이 깃허브를 위해 독립된 환경을 만듭니다):

    $ conda create -n mlbook python=3.5 anaconda
    $ source activate mlbook

이 명령은 `mlbook`이라는 이름(이름은 마음대로 바꿀 수 있습니다)으로 깨끗한 파이썬 3.5 환경을 만들고 활성화시킵니다. 이 환경은 아나콘다에 포함된 모든 과학 라이브러리를 포함시킵니다. 여기에는 텐서플로를 제외하고 우리가 필요한 모든 라이브러리가 들어 있습니다. 텐서플로는 다음과 같이 설치합니다:

    $ conda install -n mlbook -c conda-forge tensorflow=1.4.0

이 명령은 `mlbook` 환경에 텐서플로 1.4.0 버전을 설치합니다(`conda-forge` 레파지토리에서 다운로드합니다). `mlbook` 환경에 설치하지 않으려면 `-n mlbook` 옵션을 빼면 됩니다.

>(옮긴이) `conda-forge`에 텐서플로의 최신 버전이 다소 늦게 등록됩니다. 따라서 `conda`를 사용하는 것 보다는 `pip`를 사용하여 최신 버전의 텐서플로를 설치하는 것이 좋습니다. `environment.yml` 파일을 사용하여 환경을 만들었다면 자동으로 텐서플로 최신 버전이 설치됩니다.

그다음 선택적으로 주피터 확장팩을 설치할 수 있습니다. 노트북에 테이블을 표시할 때 좋지만 필수적이진 않습니다.

    $ conda install -n mlbook -c conda-forge jupyter_contrib_nbextensions

모든 것이 준비되었습니다! 이제 [주피터 시작하기](#starting-jupyter) 섹션으로 가세요.

## pip 사용하기

아나콘다를 사용하지 않는다면 이 깃허브에 필요한 파이썬 과학 라이브러리를 직접 설치해야 합니다. 특히 넘파이(NumPy), 맷플롯립(Matplotlib), 판다스(Pandas), 주피터(Jupyter) 그리고 텐서플로(TensorFlow) 등입니다. 파이썬 기본 패키징 시스템인 pip나 시스템의 패키징 시스템(가령 우분투의 apt나 맥OS의 맥포트나 홈브류)을 사용할 수 있습니다. pip를 사용하는 장점은 라이브러리와 버전이 다른 독립된 파이썬 환경을 만들기 쉽다는 것입니다(가령 이 깃허브를 위한 전용 환경). 시스템의 패키징 도구를 사용하는 장점은 파이썬 라이브러리와 시스템의 다른 패키지와 충돌할 위험이 낮다는 것입니다. 진행하는 프로젝트가 많다고 가정하고 pip를 사용하여 독립된 환경을 만들겠습니다.

pip를 사용해 필요한 라이브러리를 설치하려면 터미널에 직접 명령을 입력해야 합니다. 노트: 만약 파이썬 3가 아니고 파이썬 2를 사용한다면 이후의 모든 명령에서 `pip3`를 `pip`로, `python3`를 `python`으로 바꾸어 주세요.

먼저 최신 버전의 pip가 설치되었는지 확인합니다:

    $ pip3 install --user --upgrade pip

`--user` 옵션은 최신 버전의 pip를 현재 사용자에 대해서만 설치할 것입니다. `--user` 옵션을 빼고 시스템 전역에 걸쳐 설치하려면(즉, 모든 사용자를 위해서) 관리자 권한이 필요합니다(가령, 리눅스에서 `pip3` 대신에 `sudo pip3`를 사용합니다). `--user` 옵션을 사용하는 다음 명령들도 마찬가지입니다.

그다음 독립된 환경을 만들 수도 있습니다. 프로젝트마다 다른 라이브러리와 버전으로 구성된 환경을 만들 수 있으므로 이렇게 하는 것이 좋습니다:

    $ pip3 install --user --upgrade virtualenv
    $ virtualenv -p `which python3` env

이 명령은 현재 디렉토리에 파이썬 3 버전의 새로운 독립된 환경을 담고 있는 `env`라는 새로운 디렉토리를 만듭니다. 시스템에 파이썬 3의 버전이 여러 개라면 `` `which python3` ``을 적절한 파이썬 경로로 바꾸어 주세요.

이제 이 환경을 활성화시켜야 합니다. 환경을 활성화할 때마다 다음 명령을 실행해야 합니다.

    $ source ./env/bin/activate

다음에 pip를 사용하여 필요한 파이썬 패키지를 설치합니다. virtualenv를 사용하지 않는다면 `--user` 옵션을 사용하세요(또는 시스템 경로에 설치할 수도 있지만 아마도 관리자 권한이 필요할 것 입니다. 가령, 리눅스에서는 `pip3` 대신 `sudo pip3`를 사용합니다).

    $ pip3 install --upgrade -r requirements.txt

좋습니다! 모든 것이 설치되었으니 이제 주피터를 실행해 보죠.

## 주피터 시작하기

주피터 확장을 사용하려면(이 확장은 선택 사항으로 테이블을 미려하게 표현하기 위해 사용합니다) 먼저 관련 자바스크립트와 CSS 파일을 복사해야 합니다:

    $ jupyter contrib nbextension install --user

그런다음 "Table of Contents (2)" 확장을 활성화시킬 수 있습니다:

    $ jupyter nbextension enable toc2/main

좋습니다! 이제 주피터를 실행해 보죠:

    $ jupyter notebook

이 명령은 브라우저를 열고 현재 디렉토리 내용을 주피터의 트리 목록으로 보여줍니다. 브라우저가 자동으로 열리지 않는다면 주소 창에 [localhost:8888](http://localhost:8888/tree)를 입력하고 `index.ipynb` 파일을 클릭하세요.

노트: 주피터 확장을 활성화하고 변경하려면 [http://localhost:8888/nbextensions](http://localhost:8888/nbextensions)를 확인해 보세요.

축하합니다! 이제 머신러닝에 뛰어들 준비를 마쳤습니다!

# 기여자

유용한 피드백을 주거나 이슈를 제기하고 풀 리퀘스트를 보내 준 모든 분들에게 감사드립니다. 특히 `docker` 디렉토리를 만든 Steven Bunkley와 Ziembla에게 감사합니다.

>(옮긴이) 번역 작업을 하면서 깃허브에 포함된 `docker`를 테스트하지는 않았습니다. 윈도우즈, 리눅스, 맥OS에서 텐서플로를 쉽게 설치할 수 있기 때문에 굳이 도커를 사용할 이유는 없는 것 같습니다.
