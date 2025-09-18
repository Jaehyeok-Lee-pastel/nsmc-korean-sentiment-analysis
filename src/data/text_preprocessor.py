"""
한국어 텍스트 전처리 모듈

NSMC 프로젝트에서 사용되는 한국어 텍스트 전처리 함수들을 제공합니다.
여러 노트북에서 일관된 전처리를 위해 모듈화했습니다.

주요 기능:
- 한국어 텍스트 정제
- 데이터프레임 전처리 파이프라인
- 간단한 토큰화

작성자: 이재혁
작성일: 2025-01-17
"""

import pandas as pd
import numpy as np
import re
from tqdm import tqdm
from typing import List, Union
import logging

# 로깅 설정
logger = logging.getLogger(__name__)


def clean_korean_text(text: Union[str, float]) -> str:
    """
    한국어 텍스트 정제 함수
    
    이모티콘과 감정 표현은 보존하면서 불필요한 특수문자를 제거합니다.
    
    Args:
        text: 원본 텍스트
        
    Returns:
        str: 정제된 텍스트
        
    Examples:
        >>> clean_korean_text("ㅋㅋㅋㅋ 너무 재밌어요!!!!")
        'ㅋㅋ 너무 재밌어요!!'
        
        >>> clean_korean_text("이 영화...정말 최고입니다")
        '이 영화...정말 최고입니다'
    """
    if pd.isna(text) or text == '':
        return ''
    
    # 1. 문자열로 변환
    text = str(text)
    
    # 2. 반복 문자 정규화 (3개 이상 → 2개)
    # 이모티콘 정규화
    text = re.sub(r'([ㅋㅎㅠㅜㅡㅗㅛㅕㅑㅏㅓㅣ])\1{2,}', r'\1\1', text)
    # 특수문자 정규화
    text = re.sub(r'([!?.])\\1{2,}', r'\1\1', text)
    
    # 3. 특수문자 중 의미없는 것들 제거 (감정 표현은 보존)
    # 보존할 패턴: ㅋㅋ, ㅎㅎ, ㅠㅠ, ㅜㅜ, !!!, ???, ...
    text = re.sub(r'[^\w\sㅋㅎㅠㅜㅡㅗㅛㅕㅑㅏㅓㅣ!?.,~\-]', ' ', text)
    
    # 4. 여러 공백을 하나로
    text = re.sub(r'\s+', ' ', text)
    
    # 5. 앞뒤 공백 제거
    text = text.strip()
    
    return text


def simple_korean_tokenize(text: Union[str, float]) -> List[str]:
    """
    간단한 한국어 토큰화 함수
    (형태소 분석기 없이 공백 기준)
    
    Args:
        text: 입력 텍스트
        
    Returns:
        List[str]: 토큰 리스트
        
    Examples:
        >>> simple_korean_tokenize("안녕하세요 반갑습니다")
        ['안녕하세요', '반갑습니다']
        
        >>> simple_korean_tokenize("")
        []
    """
    if pd.isna(text) or text == '':
        return []
    
    # 공백 기준 분리
    tokens = str(text).split()
    
    # 길이 1 이상인 토큰만 보존
    tokens = [token for token in tokens if len(token) >= 1]
    
    return tokens


def preprocess_text_data(df: pd.DataFrame, 
                        text_column: str = 'document',
                        verbose: bool = True) -> pd.DataFrame:
    """
    데이터프레임의 텍스트 전처리
    
    텍스트 정제, 빈 텍스트 제거, 길이 계산을 일괄 처리합니다.
    
    Args:
        df: 입력 데이터프레임
        text_column: 텍스트 컬럼명 (기본값: 'document')
        verbose: 진행 상황 출력 여부 (기본값: True)
        
    Returns:
        DataFrame: 전처리된 데이터프레임 (cleaned_text, text_length 컬럼 추가)
        
    Examples:
        >>> df = pd.DataFrame({'document': ['좋은 영화!!', ''], 'label': [1, 0]})
        >>> result = preprocess_text_data(df)
        >>> result.columns.tolist()
        ['document', 'label', 'cleaned_text', 'text_length']
    """
    if verbose:
        print(f"📝 {len(df)}개 텍스트 전처리 시작...")
    
    # 복사본 생성
    processed_df = df.copy()
    
    # 1. 텍스트 정제
    if verbose:
        tqdm.pandas(desc="텍스트 정제")
        processed_df['cleaned_text'] = processed_df[text_column].progress_apply(clean_korean_text)
    else:
        processed_df['cleaned_text'] = processed_df[text_column].apply(clean_korean_text)
    
    # 2. 빈 텍스트 제거
    before_len = len(processed_df)
    processed_df = processed_df[processed_df['cleaned_text'].str.len() > 0]
    after_len = len(processed_df)
    
    if verbose:
        print(f"📊 빈 텍스트 제거: {before_len:,} → {after_len:,} ({before_len-after_len}개 제거)")
    
    # 3. 텍스트 길이 추가
    processed_df['text_length'] = processed_df['cleaned_text'].str.len()
    
    # 4. 기본 통계
    if verbose:
        mean_length = processed_df['text_length'].mean()
        median_length = processed_df['text_length'].median()
        print(f"📏 평균 길이: {mean_length:.1f}자")
        print(f"📏 중앙값: {median_length:.1f}자")
    
    # 인덱스 리셋
    processed_df = processed_df.reset_index(drop=True)
    
    return processed_df


def get_text_statistics(df: pd.DataFrame, 
                       text_column: str = 'cleaned_text') -> dict:
    """
    텍스트 데이터의 상세 통계 정보 반환
    
    Args:
        df: 분석할 데이터프레임
        text_column: 텍스트 컬럼명
        
    Returns:
        dict: 통계 정보 딕셔너리
    """
    text_lengths = df[text_column].str.len()
    
    stats = {
        'count': len(df),
        'mean_length': text_lengths.mean(),
        'median_length': text_lengths.median(),
        'std_length': text_lengths.std(),
        'min_length': text_lengths.min(),
        'max_length': text_lengths.max(),
        'percentile_25': text_lengths.quantile(0.25),
        'percentile_75': text_lengths.quantile(0.75),
        'empty_count': (text_lengths == 0).sum()
    }
    
    return stats


def show_preprocessing_examples(original_df: pd.DataFrame,
                              processed_df: pd.DataFrame,
                              n_examples: int = 5,
                              text_column: str = 'document') -> None:
    """
    전처리 전후 비교 예시를 출력합니다.
    
    Args:
        original_df: 원본 데이터프레임
        processed_df: 전처리된 데이터프레임
        n_examples: 출력할 예시 개수
        text_column: 원본 텍스트 컬럼명
    """
    print("📝 전처리 전후 비교:")
    print("=" * 60)
    
    for i in range(min(n_examples, len(processed_df))):
        original = original_df.iloc[i][text_column]
        cleaned = processed_df.iloc[i]['cleaned_text']
        
        print(f"\n예시 {i+1}:")
        print(f"원본: {original}")
        print(f"정제: {cleaned}")
        
        if original != cleaned:
            print("🔄 변경됨")
        else:
            print("✅ 변경없음")
    
    print("=" * 60)


# 편의 함수들
def quick_preprocess(texts: List[str]) -> List[str]:
    """
    텍스트 리스트를 빠르게 전처리합니다.
    
    Args:
        texts: 텍스트 리스트
        
    Returns:
        List[str]: 정제된 텍스트 리스트
    """
    return [clean_korean_text(text) for text in texts]


def validate_preprocessing_result(df: pd.DataFrame) -> bool:
    """
    전처리 결과를 검증합니다.
    
    Args:
        df: 전처리된 데이터프레임
        
    Returns:
        bool: 검증 통과 여부
    """
    required_columns = ['cleaned_text', 'text_length']
    
    # 필수 컬럼 확인
    for col in required_columns:
        if col not in df.columns:
            logger.error(f"필수 컬럼 누락: {col}")
            return False
    
    # 빈 텍스트 확인
    empty_texts = (df['cleaned_text'].str.len() == 0).sum()
    if empty_texts > 0:
        logger.warning(f"빈 텍스트 {empty_texts}개 발견")
    
    # 길이 일관성 확인
    calculated_lengths = df['cleaned_text'].str.len()
    length_mismatch = (calculated_lengths != df['text_length']).sum()
    
    if length_mismatch > 0:
        logger.error(f"길이 불일치: {length_mismatch}개")
        return False
    
    logger.info("전처리 결과 검증 완료")
    return True


if __name__ == "__main__":
    # 모듈 테스트
    print("🧪 text_preprocessor.py 모듈 테스트")
    
    # 테스트 데이터
    test_texts = [
        "ㅋㅋㅋㅋ 너무 재밌어요!!!!",
        "이 영화...정말 최고입니다",
        "",
        "아 진짜 별로였음 ㅠㅠㅠ"
    ]
    
    print("\n텍스트 정제 테스트:")
    for text in test_texts:
        cleaned = clean_korean_text(text)
        print(f"원본: '{text}' → 정제: '{cleaned}'")
    
    # 데이터프레임 테스트
    test_df = pd.DataFrame({
        'document': test_texts,
        'label': [1, 1, 0, 0]
    })
    
    print(f"\n데이터프레임 전처리 테스트:")
    result_df = preprocess_text_data(test_df, verbose=False)
    print(f"결과 shape: {result_df.shape}")
    print(f"컬럼: {result_df.columns.tolist()}")
    
    print("✅ 모듈 테스트 완료!")