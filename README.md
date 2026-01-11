# Soccer Object Detection Project

## 프로젝트 개요

축구 경기 영상에서 선수(players), 공(ball), 기타 객체(others)를 실시간으로 탐지하는 딥러닝 기반 객체 탐지 시스템입니다. YOLOv8 아키텍처를 활용했습니다.

## 프로젝트 목표

- 축구 경기 영상 분석 자동화
- 선수 및 공의 위치 추적을 통한 전술 분석 기반 제공
- 실시간 객체 탐지를 통한 경기 분석 플랫폼 구축

## 기술 스택

### 딥러닝 프레임워크

- **Ultralytics YOLOv8**: 최신 객체 탐지 아키텍처
- **PyTorch**: 딥러닝 백엔드

### 데이터 처리 및 시각화

- **OpenCV**: 이미지 전처리 및 후처리
- **Matplotlib**: 결과 시각화
- **NumPy**: 수치 연산

### 개발 환경

- **Python 3.x**
- **Google Colab**: GPU 기반 학습 환경
- **Google Drive**: 데이터셋 및 모델 저장소

## 데이터셋 구조

### 클래스 정의

- **Class 0**: `players` - 축구 선수
- **Class 1**: `ball` - 축구공
- **Class 2**: `others` - 기타 객체

### 데이터 분할 전략

- **Train Set**: 64% (학습용)
- **Validation Set**: 16% (검증용)
- **Test Set**: 20% (테스트용)

```
soccer_detect/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
└── soccer_detect.yaml
```

## 데이터 전처리 파이프라인

### 1. 어노테이션 변환

원본 JSON 형식의 라벨 데이터를 YOLO 형식으로 변환:

- **입력**: LabelMe JSON 형식 (절대 좌표)
- **출력**: YOLO TXT 형식 (정규화된 상대 좌표)

```
YOLO Format: <class_id> <x_center> <y_center> <width> <height>
```

### 2. Bounding Box 변환 알고리즘

```python
x_center = ((xmin + xmax) / 2) / image_width
y_center = ((ymin + ymax) / 2) / image_height
width = (xmax - xmin) / image_width
height = (ymax - ymin) / image_height
```

### 3. 데이터 증강 (YOLOv8 내장)

- Random Flip
- Random Rotation
- Color Jittering
- Mosaic Augmentation

## 모델 아키텍처

### YOLOv8 변형 실험

#### 1. YOLOv8s (Small)

- **파라미터 수**: ~11M
- **추론 속도**: 중간
- **정확도**: 높음
- **용도**: 정확도 우선 시나리오

#### 2. YOLOv8n (Nano)

- **파라미터 수**: ~3M
- **추론 속도**: 빠름
- **정확도**: 중간
- **용도**: 실시간 처리 우선 시나리오

### 학습 하이퍼파라미터

```yaml
epochs: 30
batch_size: 8
image_size: 224
optimizer: SGD (default)
learning_rate: 0.01 (default)
device: GPU (CUDA)
workers: 2
amp: False # Mixed Precision 비활성화
```

## 성능 평가 지표

### 주요 메트릭

- **mAP (mean Average Precision)**: 전체 클래스의 평균 정밀도
- **mAP@0.5**: IoU 임계값 0.5에서의 평균 정밀도
- **mAP@0.5:0.95**: IoU 임계값 0.5~0.95 범위의 평균 정밀도

### 예상 성능

YOLOv8s 모델 기준:

- mAP@0.5: ~85-90%
- mAP@0.5:0.95: ~65-75%
- FPS: 30-45 (GPU 기준)

## 주요 기능 및 구현

### 1. 데이터셋 준비

```python
# JSON to YOLO 형식 변환
# 데이터셋 분할 (Train/Valid/Test)
# YAML 설정 파일 생성
```

### 2. 모델 학습

```python
model = YOLO('yolov8s.pt')  # 사전 학습된 가중치 로드
results = model.train(
    data='soccer_detect.yaml',
    epochs=30,
    batch=8,
    imgsz=224
)
```

### 3. 모델 검증

```python
metrics = model.val(split='val')
print(f"mAP@0.5: {metrics.box.map50}")
print(f"mAP@0.5:0.95: {metrics.box.map}")
```

### 4. 추론 및 시각화

```python
# 단일 이미지 추론
results = model(image)

# Bounding Box 시각화
annotator = Annotator(image)
for box in results.boxes:
    annotator.box_label(box.xyxy[0], model.names[int(box.cls)])
```

## 프로젝트 구조

```
.
├── soccer.ipynb              # 메인 노트북
├── README.md                 # 프로젝트 문서 (본 파일)
├── data/                     # 원본 데이터셋
│   ├── fittogether.zip       # 압축된 데이터셋
│   ├── *.jpg                 # 이미지 파일
│   ├── *.json                # 원본 어노테이션
│   └── *.txt                 # 변환된 YOLO 라벨
└── soccer_detect/            # YOLO 프로젝트 폴더
    ├── train/
    ├── valid/
    ├── test/
    ├── soccer_detect.yaml    # 데이터셋 설정
    └── runs/detect/          # 학습 결과
        ├── soccerdetect_s/   # YOLOv8s 결과
        └── soccerdetect_n/   # YOLOv8n 결과
```

## 실행 방법

### 1. 환경 설정

```bash
pip install ultralytics opencv-python matplotlib pyyaml tqdm
```

### 2. 데이터셋 준비

```python
# soccer.ipynb의 셀 1-12 실행
# JSON → YOLO 변환 및 데이터 분할
```

### 3. 모델 학습

```python
# soccer.ipynb의 셀 20-21 실행
# YOLOv8s 또는 YOLOv8n 선택
```

### 4. 모델 평가 및 추론

```python
# soccer.ipynb의 셀 22-28 실행
# 검증 세트 평가 및 테스트 이미지 시각화
```

## 주요 결과

### 학습 곡선 분석

- **Loss 감소**: 안정적인 수렴 패턴
- **Validation mAP**: 에폭에 따른 지속적인 향상
- **Overfitting 방지**: 적절한 early stopping

### 추론 결과

- 선수 탐지: 높은 정확도 (90%+)
- 공 탐지: 작은 객체 탐지의 어려움 (70-80%)
- 기타 객체: 클래스 불균형으로 인한 낮은 정확도

### 시각화 예제

노트북의 마지막 셀에서 6개의 테스트 이미지에 대한 추론 결과를 확인할 수 있습니다.

## 개선 방향 및 향후 과제

### 단기 개선 사항

1. **데이터 증강 강화**

   - 다양한 조명 조건 시뮬레이션
   - 카메라 각도 변화 적용

2. **하이퍼파라미터 튜닝**

   - Learning Rate 스케줄링
   - Batch Size 최적화
   - Image Size 증가 (224 → 640)

3. **클래스 불균형 해결**
   - Weighted Loss 적용
   - 공 클래스 데이터 증강

### 중장기 발전 방향

1. **모델 앙상블**

   - 여러 YOLOv8 변형 결합
   - 추론 시간과 정확도 트레이드오프 최적화

2. **추적(Tracking) 기능 추가**

   - DeepSORT, BoT-SORT 통합
   - 선수 및 공의 궤적 추적

3. **전술 분석 시스템**
   - 선수 포메이션 분석
   - 패스 네트워크 시각화
   - 히트맵 생성

## 기술적 고려사항

- **Model Pruning**: 불필요한 가중치 제거
- **Quantization**: FP32 → INT8 변환으로 추론 속도 향상
- **Knowledge Distillation**: 큰 모델의 지식을 작은 모델로 전이
