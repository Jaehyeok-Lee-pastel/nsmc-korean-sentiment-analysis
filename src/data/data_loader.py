"""
NSMC 데이터 로딩 모듈

이 모듈은 NSMC(Naver Sentiment Movie Corpus) 데이터셋을 안전하고 효율적으로 로딩하는 
함수들을 제공합니다.

주요 기능:
- 자동 경로 탐지
- 다중 인코딩 지원 
- 데이터 검증
- 개발용 샘플링
- 상세한 로깅

작성자: 이재혁
작성일: 2025-01-17
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def find_data_directory(possible_paths: Optional[List[str]] = None) -> Optional[Path]:
    """
    NSMC 데이터 디렉토리를 자동으로 탐지합니다.
    
    Args:
        possible_paths: 탐색할 경로 리스트 (기본값: None)
        
    Returns:
        Path: 데이터 디렉토리 경로, 찾지 못하면 None
    """
    
    if possible_paths is None:
        possible_paths = [
            'data/raw',           # 프로젝트 루트에서 실행
            '../data/raw',        # notebooks 폴더에서 실행  
            '../../data/raw',     # notebooks/01_data_exploration에서 실행
            './data/raw',         # 명시적 현재 디렉토리
            '../../../data/raw'   # 더 깊은 위치에서 실행
        ]
    
    # 직접 경로 확인
    for path in possible_paths:
        if os.path.exists(path):
            data_dir = Path(path)
            logger.info(f"✅ 데이터 폴더 발견: {data_dir.absolute()}")
            return data_dir
    
    # 파일 시스템 검색
    logger.info("🔍 데이터 폴더를 직접 검색 중...")
    for root, dirs, files in os.walk('.'):
        if 'ratings_train.txt' in files:
            data_dir = Path(root)
            logger.info(f"✅ 데이터 폴더 발견: {data_dir.absolute()}")
            return data_dir
    
    return None


def load_data_with_encoding(file_path: Path, 
                          encodings: List[str] = None) -> pd.DataFrame:
    """
    여러 인코딩을 시도하여 데이터를 로딩합니다.
    
    Args:
        file_path: 로딩할 파일 경로
        encodings: 시도할 인코딩 리스트
        
    Returns:
        DataFrame: 로딩된 데이터프레임
        
    Raises:
        Exception: 모든 인코딩이 실패한 경우
    """
    
    if encodings is None:
        encodings = ['utf-8', 'cp949', 'euc-kr']
    
    for encoding in encodings:
        try:
            logger.info(f"📊 {file_path.name} 로딩 시도 (인코딩: {encoding})...")
            df = pd.read_csv(file_path, sep='\t', encoding=encoding)
            logger.info(f"✅ {file_path.name} 로딩 성공! (인코딩: {encoding})")
            return df
            
        except UnicodeDecodeError:
            logger.warning(f"❌ {encoding} 인코딩 실패, 다음 인코딩 시도...")
            continue
        except Exception as e:
            logger.error(f"❌ 데이터 로딩 실패 ({encoding}): {e}")
            continue
    
    raise Exception(f"모든 인코딩으로 {file_path.name} 로딩에 실패했습니다.")


def validate_nsmc_data(df: pd.DataFrame, data_type: str) -> bool:
    """
    NSMC 데이터의 형식을 검증합니다.
    
    Args:
        df: 검증할 데이터프레임
        data_type: 데이터 타입 ('train' 또는 'test')
        
    Returns:
        bool: 검증 통과 여부
    """
    
    logger.info(f"🔍 {data_type} 데이터 검증 중...")
    
    # 필수 컬럼 확인
    required_columns = ['id', 'document', 'label']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        logger.error(f"❌ 필수 컬럼 누락: {missing_columns}")
        return False
    
    # 데이터 크기 확인
    expected_sizes = {'train': 150000, 'test': 50000}
    if data_type in expected_sizes:
        expected_size = expected_sizes[data_type]
        actual_size = len(df)
        
        if abs(actual_size - expected_size) > expected_size * 0.01:  # 1% 오차 허용
            logger.warning(f"⚠️ {data_type} 데이터 크기 불일치: 예상 {expected_size:,}, 실제 {actual_size:,}")
    
    # 라벨 값 확인
    unique_labels = df['label'].unique()
    expected_labels = {0, 1}
    if not set(unique_labels).issubset(expected_labels):
        logger.error(f"❌ 잘못된 라벨 값: {unique_labels}")
        return False
    
    # 결측값 확인
    null_counts = df.isnull().sum()
    if null_counts.any():
        logger.warning(f"⚠️ 결측값 발견:\n{null_counts[null_counts > 0]}")
    
    logger.info(f"✅ {data_type} 데이터 검증 완료")
    return True


def get_data_summary(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """
    데이터의 기본 요약 정보를 출력합니다.
    
    Args:
        train_df: 훈련 데이터프레임
        test_df: 테스트 데이터프레임
    """
    
    print("=" * 50)
    print("📊 NSMC 데이터셋 요약")
    print("=" * 50)
    
    # 기본 정보
    total_reviews = len(train_df) + len(test_df)
    print(f"   전체 리뷰 수: {total_reviews:,}개")
    print(f"   훈련 데이터: {len(train_df):,}개 ({len(train_df)/total_reviews*100:.1f}%)")
    print(f"   테스트 데이터: {len(test_df):,}개 ({len(test_df)/total_reviews*100:.1f}%)")
    
    # 감정 분포
    print(f"\n📈 감정 분포:")
    train_pos = (train_df['label'] == 1).sum()
    train_neg = (train_df['label'] == 0).sum()
    test_pos = (test_df['label'] == 1).sum()
    test_neg = (test_df['label'] == 0).sum()
    
    print(f"   훈련 - 긍정: {train_pos:,}개 ({train_pos/len(train_df)*100:.1f}%), "
          f"부정: {train_neg:,}개 ({train_neg/len(train_df)*100:.1f}%)")
    print(f"   테스트 - 긍정: {test_pos:,}개 ({test_pos/len(test_df)*100:.1f}%), "
          f"부정: {test_neg:,}개 ({test_neg/len(test_df)*100:.1f}%)")
    
    # 샘플 데이터
    print(f"\n👀 훈련 데이터 미리보기:")
    print(train_df.head(3).to_string())
    print("=" * 50)


def load_nsmc_data(data_path: Optional[str] = None,
                   sample_size: Optional[int] = None,
                   random_state: int = 42,
                   verbose: bool = True,
                   validate: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    NSMC 데이터를 로딩합니다.
    
    Args:
        data_path: 데이터 디렉토리 경로 (기본값: 자동 탐지)
        sample_size: 로딩할 샘플 크기 (기본값: 전체 데이터)
        random_state: 샘플링 시드 (기본값: 42)
        verbose: 상세 로그 출력 여부 (기본값: True)
        validate: 데이터 검증 수행 여부 (기본값: True)
        
    Returns:
        Tuple[DataFrame, DataFrame]: (훈련 데이터, 테스트 데이터)
        
    Raises:
        FileNotFoundError: 데이터 파일을 찾을 수 없는 경우
        Exception: 데이터 로딩이 실패한 경우
    """
    
    # 로깅 레벨 조정
    if not verbose:
        logger.setLevel(logging.WARNING)
    
    # 데이터 디렉토리 찾기
    if data_path:
        data_dir = Path(data_path)
        if not data_dir.exists():
            raise FileNotFoundError(f"지정한 데이터 경로를 찾을 수 없습니다: {data_path}")
    else:
        data_dir = find_data_directory()
        if data_dir is None:
            raise FileNotFoundError(
                "NSMC 데이터 파일을 찾을 수 없습니다. "
                "data/raw/ 폴더에 ratings_train.txt와 ratings_test.txt가 있는지 확인하세요."
            )
    
    # 파일 경로 설정
    train_file = data_dir / 'ratings_train.txt'
    test_file = data_dir / 'ratings_test.txt'
    
    # 파일 존재 확인
    if not train_file.exists():
        raise FileNotFoundError(f"훈련 데이터 파일을 찾을 수 없습니다: {train_file}")
    if not test_file.exists():
        raise FileNotFoundError(f"테스트 데이터 파일을 찾을 수 없습니다: {test_file}")
    
    if verbose:
        print("📁 데이터 파일 확인:")
        print(f"   훈련 데이터: {train_file} ({train_file.stat().st_size:,} bytes)")
        print(f"   테스트 데이터: {test_file} ({test_file.stat().st_size:,} bytes)")
    
    # 데이터 로딩
    train_df = load_data_with_encoding(train_file)
    test_df = load_data_with_encoding(test_file)
    
    # 데이터 검증
    if validate:
        train_valid = validate_nsmc_data(train_df, 'train')
        test_valid = validate_nsmc_data(test_df, 'test')
        
        if not (train_valid and test_valid):
            logger.warning("⚠️ 데이터 검증 중 문제가 발견되었습니다.")
    
    # 샘플링 (개발/테스트용)
    if sample_size:
        if sample_size > len(train_df):
            logger.warning(f"⚠️ 요청된 샘플 크기({sample_size})가 전체 데이터보다 큽니다.")
        else:
            train_df = train_df.sample(n=min(sample_size, len(train_df)), 
                                     random_state=random_state).reset_index(drop=True)
            # 테스트 데이터도 비례적으로 샘플링
            test_sample_size = int(sample_size * len(test_df) / (len(train_df) + len(test_df)))
            test_df = test_df.sample(n=min(test_sample_size, len(test_df)), 
                                   random_state=random_state).reset_index(drop=True)
            
            if verbose:
                print(f"📊 샘플링 완료: 훈련 {len(train_df):,}개, 테스트 {len(test_df):,}개")
    
    # 요약 정보 출력
    if verbose:
        get_data_summary(train_df, test_df)
    
    return train_df, test_df


# 편의 함수들
def load_nsmc_sample(sample_size: int = 1000, random_state: int = 42):
    """개발용 샘플 데이터 로딩"""
    return load_nsmc_data(sample_size=sample_size, random_state=random_state)


def load_nsmc_quiet():
    """조용한 모드로 데이터 로딩"""
    return load_nsmc_data(verbose=False)


if __name__ == "__main__":
    # 모듈 테스트
    print("🧪 data_loader.py 모듈 테스트")
    try:
        train_df, test_df = load_nsmc_data()
        print("✅ 모듈 테스트 성공!")
    except Exception as e:
        print(f"❌ 모듈 테스트 실패: {e}")