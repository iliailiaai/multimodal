# multimodal

Эксперименты с мультимодальными агентами на OmniGAIA: baseline через OpenRouter и замена отдельных изображений описаниями локальной модели CapRL.

## Состав

- `OmniGAIA/` — исходный проект [RUC-NLPIR/OmniGAIA](https://github.com/RUC-NLPIR/OmniGAIA), адаптер OpenRouter, инструменты, оценка, тесты и сохранённые результаты. История исходного репозитория сохранена в истории этого проекта.
- `OmniGAIA/data/omnigaia_fixed130/` — фиксированные 130 задач для сравнения моделей; состав не меняется между экспериментами.
- `OmniGAIA/data/omnigaia_caption_smoke5/` — пять задач для технической проверки режима captioning.
- `OmniGAIA/outputs/` — снимок результатов, описаний, метрик и журналов запросов. Это архив состояния: наличие PID или `running` в старом файле не означает, что процесс сейчас работает.
- `models/` — закреплённые конфигурации, токенизаторы и процессор изображений Nemotron. Веса CapRL скачиваются отдельно.
- `11/architecture.py` — сохранённый эскиз альтернативной архитектуры.

Ключи API, виртуальные окружения, кэш медиа и веса моделей не входят в Git. При сохранении этого снимка прогон не возобновлялся.

## Протокол эксперимента

Основная модель — `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free` через OpenRouter. Сохранены также результаты `qwen/qwen3.8-omni-flash`.

В режиме `image_captioning` отдельные изображения получает только `internlm/CapRL-Qwen3VL-2B`. Её единственный текстовый запрос:

```text
Please describe this image in detail.
```

Основной модели передаётся полное описание вместо изображения. Вопрос и эталонный ответ бенчмарка captioner не получает. Видео и аудио сохраняются. Генерация описания должна завершиться EOS; обрезанные описания считаются ошибкой.

Счётчики используют локальные токенизатор и процессор Nemotron. Они сравнивают токены описания с предполагаемыми токенами изображения на фактической траектории captioning, включая повторную передачу контекста и сетевые ретраи. Это отдельная метрика от API usage и оплаты провайдера. Совпадение локального препроцессинга с API не гарантируется.

Фиксированные 130 задач выбраны по завершённым задачам прежнего Qwen-прогона. Это не случайная выборка и не оценка на полном OmniGAIA. Технические ошибки, покрытие и качество ответов учитываются отдельно.

## Подготовка окружений

Команды выполняются из корня `multimodal`:

```bash
python3.12 -m venv OmniGAIA/.venv
OmniGAIA/.venv/bin/python -m pip install -r OmniGAIA/requirements-openrouter.lock
python3.12 -m venv .venv-caption
.venv-caption/bin/python -m pip install -r OmniGAIA/requirements-caption.lock
```

Для обработки медиа нужен `ffmpeg`. Локальный captioner проверен на RTX 3060 12 GB с PyTorch 2.8.0, CUDA 12.8 и Transformers 4.57.6.

Шаблон ключей: `OmniGAIA/config/openrouter.env.example`. Заполненный файл сохранить локально как `.omnigaia-secrets.env` в корне `multimodal`; он исключён из Git. Для Reader без ключа используется `JINA_READER_MODE=anonymous`.

Докачка весов CapRL с проверкой хешей:

```bash
OmniGAIA/.venv/bin/python models/.downloads/download_caprl.py
```

Этот шаг только загружает модель и не запускает оценку. Ревизия CapRL: `b838cec8b6c4791ed6169d12db21db0b47bb70e5`. Токенизатор и процессор Nemotron взяты из [NVIDIA на Hugging Face](https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16) с ревизией `e5e9932441de940c9a62185c870ea5bcd4cd24e2`; контрольные суммы находятся в `models/Nemotron-3-Nano-Omni-tokenizer/provenance.json`.

## Проверки и запуск

Тесты без обращений к API и запуска бенчмарка:

```bash
cd OmniGAIA
.venv/bin/python -m unittest discover -s tests -q
```

Для нового эксперимента нужно явно указать новые каталоги, чтобы не смешивать его с архивом:

```bash
.venv/bin/python src/run_caption_experiment.py \
  --smoke-dir outputs/NEW_caption_smoke5 \
  --full-dir outputs/NEW_caption_fixed130
```

Эта команда запускает платные/квотируемые API-запросы: сначала пять задач smoke, затем при успешных технических проверках фиксированные 130. Все пять результатов smoke переносятся в полный прогон независимо от правильности ответов. Команда приведена для воспроизведения; автоматически при открытии или клонировании репозитория ничего не запускается.

Исходная лицензия OmniGAIA сохранена в `OmniGAIA/LICENSE`. У скачиваемых моделей и датасета действуют собственные условия использования.
