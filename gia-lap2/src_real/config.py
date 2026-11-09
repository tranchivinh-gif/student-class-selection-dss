from pathlib import Path

DATA_PATH = Path("data/data-clean/du_lieu_that.csv")
OUTPUT_DIR = Path("artifacts/src_real")
MODEL_PATH = OUTPUT_DIR / "champion.joblib"
DEPLOYED_MODEL_PATHS = {
    "baseline": OUTPUT_DIR / "baseline.joblib",
    "xgboost": OUTPUT_DIR / "xgboost.joblib",
    "catboost": OUTPUT_DIR / "catboost.joblib",
}
ID_COLUMN = "ma_hoc_sinh"
RANDOM_SEED = 42

TARGETS = [
    "Vật lí", "Hóa học", "Sinh học", "Địa lí",
    "Giáo dục kinh tế và pháp luật (GDKT&PL)",
    "Tin học", "Công nghệ", "Âm nhạc", "Mĩ thuật",
]
SUBJECT_SLUGS = [
    "van", "toan", "ngoai_ngu_1", "gdcd",
    "lich_su_dia_li", "khtn", "cong_nghe", "tin_hoc",
]
FEATURES = [
    f"L{grade}_{subject}_{period}"
    for grade in range(6, 10)
    for subject in SUBJECT_SLUGS
    for period in ("hki", "hkii", "ca_nam")
]
ANNUAL_FEATURES = [column for column in FEATURES if column.endswith("_ca_nam")]
FEATURE_SETS = {"all": FEATURES, "annual": ANNUAL_FEATURES}

# The four combinations actually offered/observed in this dataset.
ALLOWED_COMBINATIONS = (
    "000111100",  # Địa lí + GDKT&PL + Tin học + Công nghệ
    "101001100",  # Vật lí + Sinh học + Tin học + Công nghệ
    "110001100",  # Vật lí + Hóa học + Tin học + Công nghệ
    "111001000",  # Vật lí + Hóa học + Sinh học + Tin học
)

