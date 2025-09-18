# =============================================================================
# 모듈화된 함수들의 상세 동작 과정 설명
# =============================================================================

"""
📋 setup_notebook_environment() 함수 내부 동작 과정:

1. setup_project_path() 호출:
   - 현재 경로에서 'movie_review' 폴더 탐지
   - 상위 디렉토리로 올라가며 프로젝트 루트 찾기
   - data/raw 폴더 존재 여부로 검증
   - sys.path.append()로 Python 경로에 추가
   
2. setup_korean_matplotlib() 호출:
   - plt.rcParams['font.family'] = 'Malgun Gothic'
   - plt.rcParams['axes.unicode_minus'] = False
   - plt.style.use('default')
   
3. setup_seaborn_style() 호출:
   - sns.set_palette("husl") → 색상 팔레트 설정
   - sns.set_style("whitegrid") → 격자 스타일 설정
   
4. setup_pandas_options() 호출:
   - pd.set_option('display.max_columns', None)
   - pd.set_option('display.max_rows', 100)
   - pd.set_option('display.width', None)
   
5. setup_warnings() 호출:
   - warnings.filterwarnings('ignore')
   
6. check_package_versions() 호출:
   - 각 패키지의 __version__ 속성 확인
   - ImportError 및 AttributeError 예외 처리
   - 버전 정보 딕셔너리 반환 및 출력

→ 결과: 모든 환경 설정이 표준화되어 일관된 개발 환경 구성
"""

"""
📋 load_nsmc_data() 함수 내부 동작 과정:

1. find_data_directory() 호출:
   - possible_paths = ['data/raw', '../data/raw', '../../data/raw', ...]
   - for path in possible_paths: os.path.exists(path) 확인
   - 실패 시 os.walk()로 ratings_train.txt 파일 직접 검색
   
2. 파일 존재 확인:
   - train_file = data_dir / 'ratings_train.txt'
   - test_file = data_dir / 'ratings_test.txt'
   - Path.exists() 및 파일 크기(.stat().st_size) 확인
   
3. load_data_with_encoding() 호출:
   - encodings = ['utf-8', 'cp949', 'euc-kr']
   - for encoding in encodings: pd.read_csv(file, sep='\t', encoding=encoding)
   - UnicodeDecodeError 시 다음 인코딩 시도
   
4. validate_nsmc_data() 호출:
   - 필수 컬럼 ['id', 'document', 'label'] 확인
   - 데이터 크기 검증 (train: 150K, test: 50K 예상)
   - 라벨 값 {0, 1} 검증
   - 결측값 df.isnull().sum() 확인
   
5. get_data_summary() 호출:
   - 전체/훈련/테스트 데이터 개수 및 비율 계산
   - 감정 분포 (긍정/부정) 계산 및 출력
   - 샘플 데이터 head() 출력

→ 결과: 안전하게 검증된 NSMC 데이터프레임 (train_df, test_df) 반환
"""

"""
📋 preprocess_text_data() 함수 내부 동작 과정:

1. 데이터프레임 복사:
   - processed_df = df.copy() → 원본 데이터 보존
   
2. clean_korean_text() 함수를 각 행에 적용:
   a) 입력 검증: pd.isna(text) or text == '' → return ''
   b) 반복 문자 정규화:
      - re.sub(r'([ㅋㅎㅠㅜ])\\1{2,}', r'\\1\\1', text)
      - "ㅋㅋㅋㅋ" → "ㅋㅋ", "ㅠㅠㅠ" → "ㅠㅠ"
   c) 특수문자 정규화:
      - re.sub(r'([!?.])\\1{2,}', r'\\1\\1', text)
      - "!!!" → "!!", "???" → "??"
   d) 불필요한 특수문자 제거:
      - re.sub(r'[^\\w\\s이모티콘!?.,~\\-]', ' ', text)
      - 한글, 영숫자, 공백, 감정 표현만 보존
   e) 공백 정리:
      - re.sub(r'\\s+', ' ', text) → 여러 공백을 하나로
      - text.strip() → 앞뒤 공백 제거
   
3. tqdm 진행률 표시:
   - progress_apply() → 처리 진행상황 실시간 표시
   
4. 빈 텍스트 제거:
   - before_len = len(processed_df)
   - processed_df = processed_df[processed_df['cleaned_text'].str.len() > 0]
   - after_len = len(processed_df)
   - 제거된 개수 = before_len - after_len
   
5. 텍스트 길이 계산:
   - processed_df['text_length'] = processed_df['cleaned_text'].str.len()
   
6. 기본 통계 계산:
   - mean_length = processed_df['text_length'].mean()
   - median_length = processed_df['text_length'].median()
   
7. 인덱스 리셋:
   - processed_df.reset_index(drop=True)

→ 결과: 'cleaned_text', 'text_length' 컬럼이 추가된 정제된 데이터프레임
"""

"""
📋 TfidfVectorizer 동작 과정:

1. 어휘 구축 (fit 단계):
   - 모든 훈련 문서에서 토큰 추출 (공백 기준 분리)
   - min_df=2: 2개 미만 문서에 등장하는 단어 제거
   - max_df=0.95: 95% 이상 문서에 등장하는 단어 제거
   - ngram_range=(1,2): 단일 단어 + 2단어 조합 생성
   - max_features=10000: 상위 10,000개 특성만 선택
   
2. TF (Term Frequency) 계산:
   - TF(t,d) = (단어 t가 문서 d에서 등장 횟수) / (문서 d의 총 단어 수)
   - sublinear_tf=True: TF 값에 1 + log(TF) 적용
   
3. IDF (Inverse Document Frequency) 계산:
   - IDF(t) = log(전체 문서 수 / 단어 t를 포함한 문서 수)
   - 흔한 단어는 낮은 IDF, 희귀한 단어는 높은 IDF
   
4. TF-IDF 최종 계산:
   - TF-IDF(t,d) = TF(t,d) × IDF(t)
   
5. 희소 행렬 생성:
   - scipy.sparse.csr_matrix 형태로 메모리 효율적 저장
   - 대부분의 값이 0인 고차원 벡터
   
6. 정규화:
   - L2 정규화 (기본값): 각 문서 벡터의 크기를 1로 정규화

→ 결과: (문서 수 × 특성 수) 크기의 TF-IDF 희소 행렬
"""

# =============================================================================
# 실제 데이터 흐름 예시
# =============================================================================

"""
🔄 전체 데이터 처리 흐름:

1. 원본 NSMC 데이터:
   document: "ㅋㅋㅋㅋ 너무 재밌어요!!!! 완전 대박..."
   label: 1

2. clean_korean_text() 적용:
   cleaned_text: "ㅋㅋ 너무 재밌어요!! 완전 대박"
   
3. TF-IDF 토큰화:
   tokens: ["ㅋㅋ", "너무", "재밌어요", "완전", "대박", "너무 재밌어요"]
   
4. TF-IDF 벡터화:
   feature_index: {"너무": 1523, "재밌어요": 3421, "완전": 8765, ...}
   vector: [0, 0, 0, ..., 0.234, 0, ..., 0.567, ...]
   
5. 최종 결과:
   - 119,963개 훈련 문서 × 10,000개 특성
   - 99.8% 희소성 (대부분 0값)
   - 머신러닝 모델 입력 준비 완료
"""