# greenskull — прогноз даты 89M PostCrossing

Воспроизведение и улучшение метода прогнозирования миллионных вех PostCrossing, 
автор метода — Sergey Sredniy (greenskull).

## Запуск

    pip install numpy pandas scipy
    python sergei.py

## Файлы

- `sergei.py` — метод с двумя конфигурациями: `CFG` (как описано) и `CFG_IMPROVED`
- `TimeData.csv` — исходный ряд
- `gap_check.py` — анализ плотности парсинга
- `Метод Сергея — полный разбор.pdf` — презентация на 34 слайда

## Результат

- MAE: 1.62 дня (улучшенная версия)
- Прогноз 89M: 04.11.2026 20:56 UTC ± 3 дня
