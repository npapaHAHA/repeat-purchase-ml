# Repeat Purchase ML

Сервис оценивает вероятность хотя бы одного заказа в ближайшие 30 дней. Признаки считаем в SQL по покупкам клиента за предыдущие 90 дней. Готовая модель CatBoost входит в репозиторий.

## Запуск в Docker

Запустите Docker и выполните команды из корня репозитория.

~~~sh
docker build -t repeat-purchase-ml .
docker run --rm -p 127.0.0.1:8000:8000 repeat-purchase-ml
~~~

При старте контейнер читает [models/repeat-purchase.cbm](models/repeat-purchase.cbm). Когда модель загружена, `/health` возвращает `{"status":"ok"}`. Документация API открывается на [localhost:8000/docs](http://localhost:8000/docs).

В другом терминале отправьте пример запроса к `/predict`.

~~~sh
curl --fail-with-body \
  -H "Content-Type: application/json" \
  --data-binary "@example-request.json" \
  http://127.0.0.1:8000/predict
~~~

~~~powershell
curl.exe --fail-with-body -H "Content-Type: application/json" --data-binary "@example-request.json" http://127.0.0.1:8000/predict
~~~

В [примере запроса](example-request.json) у клиента три заказа на 175,50 фунта. Последний был восемь дней назад. Ниже ответ модели с округлением до шести знаков.

~~~json
{"repeat_purchase_probability": 0.420565}
~~~

Передавайте суммы в фунтах, как в исходных данных, и считайте признаки на дату прогноза. Порог для принятия решения зависит от задачи.

## Данные и оценка

Каждая строка соответствует клиенту на определённую дату. История до этой даты даёт признаки, а заказы в следующие 30 дней определяют метку. Ранние даты идут в train, следующие в validation, последние в test.

На validation сравнили модели по Average Precision. Выбранный CatBoost переобучили на train и validation, затем проверили на двух более поздних срезах, всего на 4 377 наблюдениях.

| Выборка | Модель | Average Precision | ROC-AUC |
|---|---|---:|---:|
| Validation | Логистическая регрессия | 0,593 | 0,684 |
| Validation | CatBoost | 0,613 | 0,716 |
| Test | Константный baseline | 0,396 | 0,500 |
| Test | CatBoost | **0,631** | **0,683** |

Результаты относятся к историческим данным одного магазина. Даты, размеры выборок и доли покупок есть в [отчёте](metrics.json).

## Повторить обучение

В Windows запустите PowerShell с Python 3.13.

~~~powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe download_data.py
.\.venv\Scripts\python.exe train.py
~~~

На macOS и Linux создайте окружение через `python3.13 -m venv .venv` и используйте `.venv/bin/python`.

`train.py` сохраняет модель и `metrics.json` в `artifacts/`. Чтобы запустить API с новым CatBoost, укажите путь к этому файлу модели.

~~~powershell
$env:RETAIL_MODEL_PATH = 'artifacts/model.cbm'
.\.venv\Scripts\python.exe -m uvicorn api:app
~~~


Исходные транзакции скачиваются из UCI Machine Learning Repository. Набор Daqing Chen опубликован под CC BY 4.0. [Online Retail II](https://doi.org/10.24432/C5CG6D)
