@echo off
setlocal enabledelayedexpansion

set IMAGEMAGICK_PATH="C:\Program Files\ImageMagick-7.1.1-Q16-HDRI\magick.exe"

for %%f in (*.png) do (
    set "filename=%%~nf"
    
    echo 正在处理文件: %%f
    echo 目标文件名: !filename!.jpg

    %IMAGEMAGICK_PATH% "%%f" "!filename!.jpg"
    
    if exist "!filename!.jpg" (
        echo success: !filename!.jpg
    ) else (
        echo fail: %%f
    )
    
    REM del "%%f"
)

echo 转换完成！
pause