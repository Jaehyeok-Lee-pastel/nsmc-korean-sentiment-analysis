"""
프로젝트 환경 설정 유틸리티 모듈

모든 노트북에서 공통으로 사용되는 환경 설정 함수들을 제공합니다.
일관된 개발 환경을 위해 모듈화했습니다.

주요 기능:
- 프로젝트 경로 자동 설정
- 한글 폰트 설정
- pandas/matplotlib 옵션 설정
- 패키지 버전 확인

작성자: 이재혁
작성일: 2025-01-17
"""

import sys
import os
import warnings
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Dict, Any
import logging
import numpy as np
import re
import time
import joblib

# 로깅 설정
logger = logging.getLogger(__name__)


def setup_project_path(verbose: bool = True) -> Path:
    """
    프로젝트 루트를 찾아서 Python 경로에 추가합니다.
    
    movie_review 프로젝트의 루트 디렉토리를 자동으로 탐지하고
    sys.path에 추가하여 src 모듈을 import할 수 있게 합니다.
    
    Args:
        verbose: 상세 로그 출력 여부 (기본값: True)
        
    Returns:
        Path: 프로젝트 루트 경로
        
    Raises:
        FileNotFoundError: 프로젝트 루트를 찾을 수 없는 경우
        
    Examples:
        >>> project_root = setup_project_path()
        >>> print(project_root)
        D:\dataAnalystBook\my_workspace\movie_review
    """
    current = Path.cwd()
    
    # 방법 1: 현재 경로에서 movie_review 찾기
    if 'movie_review' in str(current):
        parts = current.parts
        for i, part in enumerate(parts):
            if part == 'movie_review':
                project_root = Path(*parts[:i + 1])
                break
        else:
            project_root = current
    else:
        # 방법 2: 상위 디렉토리에서 찾기
        project_root = current
        max_depth = 5  # 무한 루프 방지
        depth = 0
        
        while project_root.parent != project_root and depth < max_depth:
            if project_root.name == 'movie_review':
                break
            if (project_root / 'data' / 'raw').exists():
                break
            project_root = project_root.parent
            depth += 1
        
        # 찾지 못한 경우 현재 경로 사용
        if depth >= max_depth:
            project_root = current
            if verbose:
                logger.warning("프로젝트 루트를 찾지 못했습니다. 현재 경로를 사용합니다.")
    
    # sys.path에 추가 (중복 방지)
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.append(project_root_str)
    
    if verbose:
        print(f"✅ 프로젝트 루트 설정: {project_root}")
    
    return project_root


def setup_korean_matplotlib(font_family: str = 'Malgun Gothic',
                           verbose: bool = True) -> bool:
    """
    한글 폰트 설정 및 matplotlib 옵션 설정
    
    Args:
        font_family: 사용할 폰트 이름 (기본값: 'Malgun Gothic')
        verbose: 상세 로그 출력 여부 (기본값: True)
        
    Returns:
        bool: 설정 성공 여부
        
    Examples:
        >>> setup_korean_matplotlib()
        ✅ 한글 폰트 설정 완료: Malgun Gothic
        True
    """
    try:
        # 한글 폰트 설정
        plt.rcParams['font.family'] = font_family
        plt.rcParams['axes.unicode_minus'] = False
        
        # 기본 스타일 설정
        plt.style.use('default')
        
        if verbose:
            print(f"✅ 한글 폰트 설정 완료: {font_family}")
        
        return True
        
    except Exception as e:
        if verbose:
            print(f"❌ 한글 폰트 설정 실패: {e}")
            print("💡 대안: 시스템에 설치된 다른 한글 폰트를 사용해보세요")
        
        return False


def setup_seaborn_style(palette: str = "husl",
                       style: str = "whitegrid",
                       verbose: bool = True) -> bool:
    """
    seaborn 스타일 설정
    
    Args:
        palette: 색상 팔레트 (기본값: "husl")
        style: 스타일 테마 (기본값: "whitegrid")
        verbose: 상세 로그 출력 여부 (기본값: True)
        
    Returns:
        bool: 설정 성공 여부
    """
    try:
        sns.set_palette(palette)
        sns.set_style(style)
        
        if verbose:
            print(f"✅ seaborn 스타일 설정 완료: {style}, {palette}")
        
        return True
        
    except Exception as e:
        if verbose:
            print(f"❌ seaborn 스타일 설정 실패: {e}")
        
        return False


def setup_pandas_options(max_columns: Optional[int] = None,
                        max_rows: int = 100,
                        width: Optional[int] = None,
                        precision: int = 3,
                        verbose: bool = True) -> bool:
    """
    pandas 표시 옵션 설정
    
    Args:
        max_columns: 최대 컬럼 수 (기본값: None - 전체 표시)
        max_rows: 최대 행 수 (기본값: 100)
        width: 표시 너비 (기본값: None - 자동)
        precision: 소수점 자릿수 (기본값: 3)
        verbose: 상세 로그 출력 여부 (기본값: True)
        
    Returns:
        bool: 설정 성공 여부
    """
    try:
        pd.set_option('display.max_columns', max_columns)
        pd.set_option('display.max_rows', max_rows)
        pd.set_option('display.width', width)
        pd.set_option('display.precision', precision)
        
        if verbose:
            print(f"✅ pandas 옵션 설정 완료: 최대 {max_rows}행, {max_columns or '전체'}컬럼 표시")
        
        return True
        
    except Exception as e:
        if verbose:
            print(f"❌ pandas 옵션 설정 실패: {e}")
        
        return False


def setup_warnings(action: str = 'ignore',
                  verbose: bool = True) -> bool:
    """
    경고 메시지 설정
    
    Args:
        action: 경고 처리 방식 ('ignore', 'default', 'error' 등)
        verbose: 상세 로그 출력 여부 (기본값: True)
        
    Returns:
        bool: 설정 성공 여부
    """
    try:
        warnings.filterwarnings(action)
        
        if verbose:
            if action == 'ignore':
                print("✅ 경고 메시지 숨김 설정 완료")
            else:
                print(f"✅ 경고 메시지 설정 완료: {action}")
        
        return True
        
    except Exception as e:
        if verbose:
            print(f"❌ 경고 설정 실패: {e}")
        
        return False


def check_package_versions(packages: Optional[list] = None,
                          verbose: bool = True) -> Dict[str, str]:
    """
    주요 패키지 버전 확인
    
    Args:
        packages: 확인할 패키지 리스트 (기본값: None - 주요 패키지들)
        verbose: 버전 정보 출력 여부 (기본값: True)
        
    Returns:
        Dict[str, str]: 패키지명-버전 딕셔너리
    """
    if packages is None:
        packages = [
            'pandas', 'numpy', 'matplotlib', 'seaborn', 
            'scikit-learn', 'tqdm'
        ]
    
    versions = {}
    
    for package in packages:
        try:
            if package == 'scikit-learn':
                import sklearn
                version = sklearn.__version__
                versions['scikit-learn'] = version
            else:
                module = __import__(package)
                version = module.__version__
                versions[package] = version
                
            if verbose:
                print(f"📦 {package}: {version}")
                
        except ImportError:
            versions[package] = "Not installed"
            if verbose:
                print(f"❌ {package}: 설치되지 않음")
        except AttributeError:
            versions[package] = "Version unknown"
            if verbose:
                print(f"⚠️ {package}: 버전 정보 없음")
    
    return versions


def setup_notebook_environment(font_family: str = 'Malgun Gothic',
                              seaborn_palette: str = "husl",
                              pandas_max_rows: int = 100,
                              ignore_warnings: bool = True,
                              check_versions: bool = True,
                              verbose: bool = True) -> Dict[str, Any]:
    """
    노트북 환경을 일괄 설정합니다.
    
    이 함수 하나로 모든 기본 설정을 완료할 수 있습니다.
    
    Args:
        font_family: 한글 폰트 이름
        seaborn_palette: seaborn 색상 팔레트
        pandas_max_rows: pandas 최대 표시 행 수
        ignore_warnings: 경고 숨김 여부
        check_versions: 패키지 버전 확인 여부
        verbose: 상세 로그 출력 여부
        
    Returns:
        Dict[str, Any]: 설정 결과 및 정보
        
    Examples:
        >>> setup_notebook_environment()
        ✅ 프로젝트 루트 설정: D:\...\movie_review
        ✅ 한글 폰트 설정 완료: Malgun Gothic
        ✅ seaborn 스타일 설정 완료: whitegrid, husl
        ✅ pandas 옵션 설정 완료: 최대 100행, 전체컬럼 표시
        ✅ 경고 메시지 숨김 설정 완료
        📦 pandas: 1.5.3
        ...
    """
    results = {}
    
    if verbose:
        print("🔧 노트북 환경 설정 시작...")
        print("=" * 50)
    
    # 1. 프로젝트 경로 설정
    try:
        project_root = setup_project_path(verbose=verbose)
        results['project_root'] = str(project_root)
        results['path_setup'] = True
    except Exception as e:
        results['path_setup'] = False
        if verbose:
            print(f"❌ 프로젝트 경로 설정 실패: {e}")
    
    # 2. 한글 폰트 설정
    results['font_setup'] = setup_korean_matplotlib(font_family, verbose=verbose)
    
    # 3. seaborn 스타일 설정
    results['seaborn_setup'] = setup_seaborn_style(seaborn_palette, verbose=verbose)
    
    # 4. pandas 옵션 설정
    results['pandas_setup'] = setup_pandas_options(max_rows=pandas_max_rows, verbose=verbose)
    
    # 5. 경고 설정
    if ignore_warnings:
        results['warnings_setup'] = setup_warnings('ignore', verbose=verbose)
    else:
        results['warnings_setup'] = True
    
    # 6. 패키지 버전 확인
    if check_versions:
        if verbose:
            print("\n📋 설치된 패키지 버전:")
        results['versions'] = check_package_versions(verbose=verbose)
    
    if verbose:
        print("=" * 50)
        success_count = sum(1 for k, v in results.items() 
                          if k.endswith('_setup') and v)
        print(f"✅ 환경 설정 완료! ({success_count}/5 성공)")
    
    return results


# 편의 함수들
def quick_setup():
    """빠른 환경 설정 (로그 최소화)"""
    return setup_notebook_environment(verbose=False)


def check_setup_status() -> bool:
    """현재 설정 상태 확인"""
    try:
        # 프로젝트 경로 확인
        if str(Path.cwd()) not in sys.path:
            return False
            
        # 한글 폰트 확인
        if plt.rcParams['font.family'] == ['DejaVu Sans']:
            return False
            
        return True
        
    except Exception:
        return False


if __name__ == "__main__":
    # 모듈 테스트
    print("🧪 setup.py 모듈 테스트")
    print("=" * 40)
    
    # 전체 환경 설정 테스트
    results = setup_notebook_environment()
    
    print(f"\n📊 설정 결과:")
    for key, value in results.items():
        if key.endswith('_setup'):
            status = "✅" if value else "❌"
            print(f"  {key}: {status}")
    
    print("\n✅ 모듈 테스트 완료!")