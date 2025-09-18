"""
TF-IDF 벡터화 모듈

현재 노트북에서 사용하고 있는 TF-IDF 설정을 모듈화했습니다.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


def create_tfidf_vectorizer():
    """
    현재 사용 중인 기본 TF-IDF 벡터라이저를 생성합니다.
    
    현재 text_cleaning.ipynb에서 사용하는 설정과 동일:
    - max_features=10000
    - min_df=2  
    - max_df=0.95
    - ngram_range=(1, 2)
    - sublinear_tf=True
    """
    
    tfidf = TfidfVectorizer(
        max_features=10000,      # 최대 특성 수
        min_df=2,                # 최소 문서 빈도
        max_df=0.95,             # 최대 문서 빈도 (너무 흔한 단어 제거)
        ngram_range=(1, 2),      # 1-gram, 2-gram 사용
        sublinear_tf=True        # 서브리니어 TF 스케일링
    )
    
    return tfidf


def show_tfidf_stats(X_tfidf, feature_names):
    """
    TF-IDF 벡터화 결과 통계를 출력합니다.
    
    현재 노트북에서 출력하는 것과 동일한 정보를 제공합니다.
    """
    
    print(f"특성 수: {X_tfidf.shape[1]:,}개")
    print(f"희소성: {(1 - X_tfidf.nnz / np.prod(X_tfidf.shape)) * 100:.1f}%")
    print(f"주요 특성 예시 (처음 20개):")
    print(feature_names[:20])


if __name__ == "__main__":
    # 간단한 테스트
    tfidf = create_tfidf_vectorizer()
    print("TF-IDF 벡터라이저 생성 완료")