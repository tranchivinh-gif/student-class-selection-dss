@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo Dang khoi dong giao dien thu nghiem K-Means...
echo Khi giao dien san sang, trinh duyet se mo tai http://localhost:8766
echo De dung giao dien, dong cua so nay hoac nhan Ctrl+C.
echo.

python -m streamlit run src\app_kmeans.py --server.port 8766

if errorlevel 1 (
    echo.
    echo Khong the khoi dong. Hay kiem tra Python va cai thu vien bang:
    echo python -m pip install -r requirements.txt
    pause
)
