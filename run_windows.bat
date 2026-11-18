@echo off
cd /d "%~dp0"
echo =============================================
echo Dean's Dilemma - Streamlit Analytics Dashboard
echo =============================================
where py >nul 2>nul
if %errorlevel%==0 (set PY=py) else (set PY=python)
%PY% -m pip install --upgrade pip
%PY% -m pip install -r requirements.txt
if errorlevel 1 goto error
 echo.
echo Starting Streamlit...
%PY% -m streamlit run app.py
pause
goto end
:error
echo Installation failed. Check Python and internet connection.
pause
:end
