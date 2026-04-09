"""Test configuration for recsys-fastapi.

Đảm bảo package `app` luôn import được và DB được reset giữa các test.
"""

import sys
from pathlib import Path

import pytest


# Thư mục gốc của service recsys-fastapi (folder chứa package `app`).
ROOT_DIR = Path(__file__).resolve().parent.parent

# Thêm vào sys.path nếu chưa có.
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from app.db.session import Base, engine  # noqa: E402  (import sau khi chỉnh sys.path)


@pytest.fixture(autouse=True)
def reset_db():
    """Drop & create lại toàn bộ schema trước mỗi test.

    Điều này giúp tránh lỗi UNIQUE constraint giữa các test khi dùng cùng SQLite file.
    """

    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    yield
