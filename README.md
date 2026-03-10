<p align="center">
<img src="static/images/banner.png" alt="drawing" width="600"/>
</p>

# TalkToModel: 대화형 자연어 인터랙션으로 머신러닝 모델을 설명하기

[![Python application](https://github.com/dylan-slack/TalkToModel/actions/workflows/python-app.yml/badge.svg)](https://github.com/dylan-slack/TalkToModel/actions/workflows/python-app.yml) [![arXiv](https://img.shields.io/badge/arXiv-2207.04154-b31b1b.svg)](https://arxiv.org/abs/2207.04154)


[TalkToModel 논문](https://arxiv.org/abs/2207.04154) 페이지에 오신 것을 환영합니다! 이 프로젝트의 목표는 누구나 자연어 대화를 통해 학습된 머신러닝 모델의 예측을 이해할 수 있도록 하는 것입니다. 궁극적으로 이 프로젝트는 **대화형 XAI(conversational XAI)** 플랫폼입니다!

<p align="center">
&nbsp;&nbsp;&nbsp;
<img src="static/images/output.gif" alt="drawing" width="600"/>
</p>

## 링크

- [arXiv TalkToModel 논문](https://arxiv.org/abs/2207.04154) 보기 ✨
- [GitHub 코드](https://github.com/dylan-slack/TalkToModel) 보기 🖥️
- 당뇨병 예측 과제에서 [TalkToModel 데모](https://nlp.ics.uci.edu/talk-to-healthcare-model/)를 운영 중입니다 🚀

이 연구가 도움이 되었다면, 아래와 같이 인용 부탁드립니다!

```bibtex
@Article{Slack2023,
author={Slack, Dylan
and Krishna, Satyapriya
and Lakkaraju, Himabindu
and Singh, Sameer},
title={Explaining machine learning models with interactive natural language conversations using TalkToModel},
journal={Nature Machine Intelligence},
year={2023},
month={Jul},
day={27},
abstract={Practitioners increasingly use machine learning (ML) models, yet models have become more complex and harder to understand. To understand complex models, researchers have proposed techniques to explain model predictions. However, practitioners struggle to use explainability methods because they do not know which explanation to choose and how to interpret the explanation. Here we address the challenge of using explainability methods by proposing TalkToModel: an interactive dialogue system that explains ML models through natural language conversations. TalkToModel consists of three components: an adaptive dialogue engine that interprets natural language and generates meaningful responses; an execution component that constructs the explanations used in the conversation; and a conversational interface. In real-world evaluations, 73{\%} of healthcare workers agreed they would use TalkToModel over existing systems for understanding a disease prediction model, and 85{\%} of ML professionals agreed TalkToModel was easier to use, demonstrating that TalkToModel is highly effective for model explainability.},
issn={2522-5839},
doi={10.1038/s42256-023-00692-8},
url={https://doi.org/10.1038/s42256-023-00692-8}
}
```

[업데이트] 본 연구는 NeurIPS TSRML 워크숍에서 Honorable Mention Outstanding Paper를 수상했습니다 🎉

또한, 이 연구의 동기가 된 도메인 전문가의 모델 이해 요구를 다룬 [선행 논문](https://arxiv.org/abs/2202.01875)도 작성했습니다. 제목은 *Rethinking Explainability as a Dialogue: A Practitioner's Perspective*이며, 함께 읽어보시길 권장드립니다!


## 목차

- [개요](#개요)
- [설치](#설치)
- [TalkToModel 애플리케이션 실행](#talktomodel-애플리케이션-실행)
- [내 모델과 데이터셋으로 실행](#내-모델과-데이터셋으로-실행)
- [실험](#실험)
- [개발](#개발)

## 개요

아래는 시스템의 목적과 범위에 대한 간단한 설명입니다.

### 목적

머신러닝 모델이 일상생활 전반에 점점 더 통합되면서, 누구나 모델과 상호작용하고 이를 이해할 수 있도록 하는 것이 중요해졌습니다. TalkToModel은 이러한 목표를 실현하며, *누구나* 머신러닝 모델과 채팅하면서 모델의 예측을 이해할 수 있게 해줍니다.

시스템의 동작 방식과 더 자세한 배경은 [논문](https://arxiv.org/abs/2207.04154)을 참고해주세요.

### 범위

TalkToModel은 *테이블형(tabular)* 모델과 데이터셋을 지원합니다. 예를 들어, 현재 형태의 시스템에서는 대출 예측 과제로 학습한 랜덤 포레스트와는 대화할 수 있지만, 감성 분석 과제로 학습한 BERT와는 사용할 수 없습니다.

## 설치

TalkToModel을 실행하려면 conda 환경을 구성하거나 Docker를 사용해 Flask 앱을 직접 실행할 수 있습니다.

참고로 GPU 추론을 위해서는 **CUDA 11.3**이 필요합니다. CUDA 11.3이 없는 상태에서 Docker 애플리케이션을 실행하면 크래시가 발생합니다. 이 경우 CPU 추론으로 전환하면(다소 느리지만) 실행할 수 있습니다([안내](#gpu-가용성)).

### Docker

Docker를 사용할 경우 이 설정 단계는 건너뛰어도 됩니다 ⏭️

### Conda

환경을 만들고 의존성을 설치합니다.

```shell
conda create -n ttm python=3.9
conda activate ttm
```

요구사항 설치:

```shell
pip install -r requirements.txt
```

잘하셨습니다 👍

## 사전 구성된 TalkToModel 데모 실행

이제 [논문](https://arxiv.org/abs/2207.04154)에서 사용한 데이터셋/모델 중 하나로 Flask 애플리케이션을 실행하는 방법을 설명합니다. 당뇨병, 범죄, 신용 예측 과제가 포함되어 있습니다.

### 설정

여기서는 데모와 파싱 모델 선택 방법을 다룹니다. 당뇨병 데이터셋 데모를 바로 실행하고 싶다면 [Conda로 실행](#conda로-실행) 또는 [Docker로 실행](#docker로-실행)으로 이동하세요.

#### 데모 및 파싱 모델 선택
이 소프트웨어는 [gin-config](https://github.com/google/gin-config)로 설정합니다. 전역 파라미터는 `./global_config.gin`에, 데모별 파라미터는 `./configs` 디렉터리에 저장되어 있습니다. 예: 당뇨병 예측 데모는 `./configs/diabetes-config.gin`.

이 저장소는 기본적으로 당뇨병 예측 과제를 위해 파인튜닝된 `t5-small` 모델이 설정되어 있으며, `./global_config.gin`을 수정해 변경할 수 있습니다:

```shell
GlobalArgs.config = "./configs/{demo}-config.gin"
```

그리고 `{demo}`를 `diabetes`, `compas`, `german` 중 하나로 바꾸면 됩니다. 또한 Hugging Face Hub에 최적의 파인튜닝 `t5-small` 및 `t5-large` 파싱 모델을 제공합니다. 이는 `./configs/{demo}-config.gin`을 수정해 선택할 수 있습니다:

```shell
# t5 small 모델
ExplainBot.parsing_model_name = "ucinlp/{demo}-t5-small"
# t5 large 모델
ExplainBot.parsing_model_name = "ucinlp/{demo}-t5-large"
```

선택한 파싱 모델은 Hub에서 자동으로 다운로드됩니다.

#### GPU 가용성

기본적으로 시스템은 모델을 CUDA 디바이스로 올리려고 시도하며, 사용 가능한 디바이스가 없으면 크래시가 발생합니다. CPU 추론으로 전환하거나 다른 CUDA 디바이스를 사용하려면 `./parsing/t5/gin_configs/t5-large.gin`을 수정하세요.

```shell
load_t5_params.device = "{device}"
```

여기서 `{device}`는 원하는 디바이스입니다(예: `cpu`).

### Conda로 실행

Conda 환경을 설치했다면, [Flask](https://flask.palletsprojects.com/en/2.2.x/) 웹 앱은 아래 명령으로 실행할 수 있습니다.
```shell
python flask_app.py
```

### Docker로 실행

[Docker](https://www.docker.com)로 실행하려면 먼저 Docker 앱을 빌드합니다.

```shell
sudo docker build -t ttm .
```

그 다음 이미지를 실행합니다.

```shell
sudo docker run -d -p 4000:4000 ttm
```

### 사용하기

애플리케이션을 빌드하는 데 1~2분 정도 걸릴 수 있습니다. 이는 실시간 사용자 경험을 개선하기 위해 사전에 꽤 많은 계산(주로 설명값)을 캐시하기 때문입니다. 참고로 M1 MacBook Pro 기준, 처음부터 빌드하면 약 5분 정도 소요됩니다. 다만 데모를 한 번 실행하고 나면 해당 계산 결과가 `./cache` 폴더에 저장되므로, 이후 시작 속도는 매우 빨라집니다.

이제 끝입니다! 앱이 실행 중이어야 합니다 💥

## 실험

여기서는 논문의 실험 실행 방법을 설명합니다. 실험을 위해서는 각 `diabetes`, `german`, `compas` gin 설정 파일에서 아래 값을 설정하세요.

```
ExplainBot.skip_prompts = False
```

이렇게 해야 시스템이 각 데이터셋에 대한 프롬프트를 생성합니다. 파인튜닝된 데모 파싱 모델의 경우 이 단계가 불필요하고 시작 시간을 크게 줄여주기 때문에 기본값을 `True`로 두었습니다.

### 모델 파인튜닝

파싱 모델을 파인튜닝하려면 아래 명령을 실행하세요.

```shell
python parsing/t5/start_fine_tuning.py --gin parsing/t5/gin_configs/t5-{small, base, large}.gin --dataset {diabetes, german, compas}
```

여기서 `{small, base, large}`와 `{diabetes, german, compas}`는 각각 가능한 값 중 하나입니다. 참고로 이 실험은 학습 추적 및 최적 검증 모델 관리를 위해 [Weights & Biases](https://wandb.ai/site) 사용이 필요합니다.

편의를 위해 최적 검증 사전학습 모델들도 Hugging Face에서 다운로드할 수 있도록 제공하고 있습니다: [https://huggingface.co/dslack/all-finetuned-ttm-models](https://huggingface.co/dslack/all-finetuned-ttm-models). 제공된 zip 파일을 내려받아 모델을 `./parsing/t5/models`에 압축 해제하면 됩니다.

### 정답 파싱 정확도 평가

모든 모델을 다운로드한 뒤, 아래 명령으로 파싱 정확도를 계산할 수 있습니다.

```shell
python experiments/generate_parsing_results.py
```

결과는 `./experiments/results_store` 디렉터리에 저장됩니다.

## 내 모델과 데이터셋으로 실행

단계별 가이드는 `./tutorials/running-on-your-own-model.ipynb` 튜토리얼을 참고하세요. 대화 설정을 위한 다양한 옵션도 함께 설명되어 있습니다.

## 개발

TalkToModel은 새로운 기능을 비교적 쉽게 확장할 수 있습니다. 시스템 확장 방법은 `./tutorials/extending-ttm.md`를 참고하세요.

## 테스트

저장소 루트 디렉터리에서 `pytest`를 실행하면 테스트를 수행할 수 있습니다.

## 인용

인용해주세요 🫶

```bibtex
@Article{Slack2023,
author={Slack, Dylan
and Krishna, Satyapriya
and Lakkaraju, Himabindu
and Singh, Sameer},
title={Explaining machine learning models with interactive natural language conversations using TalkToModel},
journal={Nature Machine Intelligence},
year={2023},
month={Jul},
day={27},
abstract={Practitioners increasingly use machine learning (ML) models, yet models have become more complex and harder to understand. To understand complex models, researchers have proposed techniques to explain model predictions. However, practitioners struggle to use explainability methods because they do not know which explanation to choose and how to interpret the explanation. Here we address the challenge of using explainability methods by proposing TalkToModel: an interactive dialogue system that explains ML models through natural language conversations. TalkToModel consists of three components: an adaptive dialogue engine that interprets natural language and generates meaningful responses; an execution component that constructs the explanations used in the conversation; and a conversational interface. In real-world evaluations, 73{\%} of healthcare workers agreed they would use TalkToModel over existing systems for understanding a disease prediction model, and 85{\%} of ML professionals agreed TalkToModel was easier to use, demonstrating that TalkToModel is highly effective for model explainability.},
issn={2522-5839},
doi={10.1038/s42256-023-00692-8},
url={https://doi.org/10.1038/s42256-023-00692-8}
}
```

## 문의

문의 사항이나 실행 중 이슈가 있다면 [dslack@uci.edu](dslack@uci.edu)로 연락해주세요.

특히 사용자 고유의 모델/데이터에서 실행하려는 경우에도 언제든 문의하거나 이슈를 등록해 주세요. 기꺼이 도와드리겠습니다 ❤️!
