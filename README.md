# 🎬 NSMC 기반 한국어 영화 리뷰 감정분석

## 📋 프로젝트 개요

네이버 영화 리뷰 데이터(NSMC)를 활용한 한국어 감정분석 프로젝트입니다. 체계적인 ML 파이프라인을 구축하여 포트폴리오급 완성도를 목표로 합니다.

### 🎯 프로젝트 목표

- **정확도 목표**: NSMC 테스트 데이터 기준 **85% 이상** 정확도 달성
- **기술적 목표**: 체계적인 ML 파이프라인 구축 및 다양한 모델 비교
- **학습 목표**: 한국어 NLP 실무 경험 및 전문적 개발 프로세스 체득
- **포트폴리오**: 기업급 코드 품질 및 문서화 완성

### 📊 데이터셋 정보

| 항목 | 상세 정보 |
|------|-----------|
| **출처** | [NSMC (Naver Sentiment Movie Corpus)](https://github.com/e9t/nsmc) |
| **총 규모** | 200,000개 영화 리뷰 |
| **훈련 데이터** | 150,000개 리뷰 |
| **테스트 데이터** | 50,000개 리뷰 |
| **라벨 구성** | 긍정(1) / 부정(0) - 완벽한 50:50 균형 |
| **데이터 형식** | TSV (탭 구분) 텍스트 파일 |

### 🛠️ 기술 스택

**개발 환경**
- Python 3.9
- Jupyter Notebook
- Git & GitHub

**데이터 처리**
- pandas, numpy
- matplotlib, seaborn

**자연어 처리**
- konlpy (한국어 형태소 분석)
- scikit-learn (전통적 ML)
- transformers (KoBERT, KoELECTRA)

**머신러닝**
- 전통적 ML: 로지스틱 회귀, SVM, Random Forest
- 딥러닝: LSTM, CNN
- Transformer: KoBERT, KoELECTRA

## 🚀 빠른 시작

### 환경 설정

```bash
# 1. 저장소 클론
git clone https://github.com/Jaehyeok-Lee-pastel/nsmc-korean-sentiment-analysis.git
cd nsmc-korean-sentiment-analysis

# 2. 가상환경 생성 및 활성화
conda create -n nsmc_sentiment python=3.9 -y
conda activate nsmc_sentiment

# 3. 필요 패키지 설치
pip install -r requirements.txt

# 4. Jupyter Notebook 실행
jupyter notebook
```

### 데이터 준비

```bash
# NSMC 데이터 다운로드 (data/raw/ 폴더에 저장)
wget -P data/raw/ https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt
wget -P data/raw/ https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt
```

## 📁 프로젝트 구조

```
nsmc-korean-sentiment-analysis/
│
├── 📊 data/                           # 데이터 저장소
│   ├── raw/                          # 원본 NSMC 데이터
│   ├── processed/                    # 전처리된 데이터
│   ├── interim/                      # 중간 처리 결과
│   └── external/                     # 외부 데이터
│
├── 📓 notebooks/                      # 분석 노트북
│   ├── 01_data_exploration/          # 데이터 탐색
│   ├── 02_data_preprocessing/        # 데이터 전처리
│   ├── 03_feature_engineering/       # 특성 공학
│   ├── 04_modeling/                  # 모델링
│   └── 05_evaluation/                # 평가 및 해석
│
├── 🐍 src/                           # 소스 코드
│   ├── data/                         # 데이터 처리 모듈
│   ├── features/                     # 특성 추출 모듈
│   ├── models/                       # 모델 관련 모듈
│   ├── utils/                        # 유틸리티 함수
│   └── visualization/                # 시각화 모듈
│
├── 🤖 models/                        # 학습된 모델
├── 📊 reports/                       # 보고서 및 시각화
├── 📈 results/                       # 분석 결과
├── ⚙️ config/                        # 설정 파일
└── 🧪 tests/                         # 테스트 코드
```

## 📈 모델 개발 로드맵

### Phase 1: 기초 분석 (Week 1-2)
- [x] 환경 설정 및 프로젝트 구조 생성
- [x] GitHub 저장소 설정
- [ ] NSMC 데이터 탐색 및 특성 분석
- [ ] 기본 통계 분석 및 시각화

### Phase 2: 전통적 ML (Week 3-4)
- [ ] 한국어 텍스트 전처리 파이프라인
- [ ] TF-IDF, Bag of Words 특성 추출
- [ ] 로지스틱 회귀, SVM 모델 구현
- [ ] **목표**: 75% 이상 정확도 달성

### Phase 3: 딥러닝 (Week 5-6)
- [ ] Word2Vec, FastText 임베딩
- [ ] LSTM, CNN 모델 구현
- [ ] **목표**: 80% 이상 정확도 달성

### Phase 4: Transformer (Week 7-8)
- [ ] KoBERT, KoELECTRA 파인튜닝
- [ ] 모델 해석 및 어텐션 시각화
- [ ] **목표**: 85% 이상 정확도 달성

### Phase 5: 완성 (Week 9-10)
- [ ] 웹 데모 개발 (Streamlit)
- [ ] 종합 성능 비교 및 분석
- [ ] 기술 문서 및 포트폴리오 완성

## 📊 현재 진행 상황

- [x] **환경 설정 완료** - 가상환경 및 패키지 설치
- [x] **프로젝트 구조 생성** - 체계적인 폴더 구조 및 Git 설정
- [x] **GitHub 연동** - 원격 저장소 연결 및 초기 커밋
- [ ] **데이터 탐색** - NSMC 데이터 분석 및 특성 파악
- [ ] **데이터 전처리** - 한국어 텍스트 정제 파이프라인
- [ ] **기본 모델링** - 전통적 ML 알고리즘 구현
- [ ] **딥러닝 모델** - LSTM, CNN 구현
- [ ] **Transformer 모델** - KoBERT 파인튜닝
- [ ] **웹 데모** - Streamlit 기반 감정분석 서비스
- [ ] **문서화** - 기술 블로그 및 발표자료

## 🔍 주요 특징

### 한국어 NLP 특화
- 한국어 형태소 분석 (KoNLPy)
- 구어체 및 인터넷 슬랭 처리
- 띄어쓰기 및 맞춤법 오류 대응

### 체계적 개발 프로세스
- 단계별 노트북으로 분석 과정 추적
- 모듈화된 코드 구조로 재사용성 확보
- Git을 통한 체계적 버전 관리

### 다양한 모델 비교
- 전통적 ML부터 최신 Transformer까지
- 성능 지표별 상세 비교 분석
- 모델 해석 및 개선 방안 제시

## 📚 참고 자료

- [NSMC 원본 저장소](https://github.com/e9t/nsmc)
- [KoBERT](https://github.com/SKTBrain/KoBERT)
- [KoNLPy 문서](https://konlpy.org/ko/latest/)
- [Scikit-learn 감정분석 가이드](https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html)

## 🤝 기여하기

이 프로젝트는 학습 목적으로 제작되었습니다. 개선 사항이나 질문이 있으시면 Issues를 통해 의견을 공유해주세요.

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

## 📞 연락처

- **GitHub**: [@Jaehyeok-Lee-pastel](https://github.com/Jaehyeok-Lee-pastel)
- **이메일**: [프로젝트 관련 문의]
- **블로그**: [기술 블로그 주소]

---

**⭐ 이 프로젝트가 도움이 되셨다면 Star를 눌러주세요!**
