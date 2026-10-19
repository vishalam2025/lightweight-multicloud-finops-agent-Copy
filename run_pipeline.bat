@echo off
setlocal

cd /d "%~dp0"

if not exist logs mkdir logs
if not exist results mkdir results

set "DOCKER_EXE=C:\Program Files\Docker\Docker\resources\bin\docker.exe"
set "LOG_FILE=%cd%\logs\scheduled_pipeline.log"

echo. >> "%LOG_FILE%"
echo ===== Run started: %date% %time% ===== >> "%LOG_FILE%"

"%DOCKER_EXE%" info >nul 2>&1
if errorlevel 1 (
    echo ERROR: Docker engine is unavailable. >> "%LOG_FILE%"
    exit /b 1
)

"%DOCKER_EXE%" run --rm --cpus=1 --memory=1g --mount "type=bind,source=%cd%\data,target=/app/data,readonly" --mount "type=bind,source=%cd%\results,target=/app/results" finops-da3 >> "%LOG_FILE%" 2>&1

set "RUN_STATUS=%errorlevel%"
echo ===== Run finished: %date% %time% Exit code: %RUN_STATUS% ===== >> "%LOG_FILE%"

exit /b %RUN_STATUS%