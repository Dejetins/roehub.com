# Indicators — Registry + YAML Defaults (v1)

Этот документ фиксирует архитектуру и контракты для **IND-EPIC-02 — Registry (code defs + YAML defaults)** в bounded context `indicators`.

Цель EPIC:

* UI должен видеть **полную библиотеку индикаторов** (hard bounds, типы параметров, входные серии, layout);
* UI должен видеть **дефолтные диапазоны/значения** (defaults), зависящие от окружения (`dev/prod/test`);
* система должна **fail-fast** на старте, если YAML некорректен (выходит за hard bounds, шаг не согласован, неизвестный индикатор/параметр и т.д.);
* API `GET /indicators` отдаёт registry (hard bounds + UI defaults).

Дополнительно фиксируем совместимость: **сохранение выбранных комбинаций индикаторов** (explicit params) и детерминированный `variant_key`, привязанный к `instrument_id` и `timeframe`.

---

## Scope / Non-goals

### In scope (EPIC-02)

1. Code-level definitions индикаторов по группам:

* `src/trading/contexts/indicators/domain/definitions/{ma,trend,volatility,momentum,volume,structure}.py`

2. YAML defaults по окружениям:

* `configs/dev/indicators.yaml`
* `configs/prod/indicators.yaml`
* `configs/test/indicators.yaml`

3. Loader + validator defaults (fail-fast) и “merge” hard defs + UI defaults.
4. API endpoint:

* `apps/api/routes/indicators.py` + wiring в `apps/api/wiring/...`

### Out of scope (EPIC-02)

* compute engine (numba/numpy), staged backtest, оптимизация, хранение результатов — это следующие EPIC’и;
* persistence профилей/сохранённых наборов — отдельный EPIC (будет опираться на `variant_key` v1 из этого документа).

---

## Source of Truth

**Hard bounds и структура индикаторов** — в code defs (`domain/definitions/*`).
**UI defaults** — в YAML (`configs/*/indicators.yaml`) и НЕ могут расширять hard bounds, только сужать/задавать subset.

Публичный слой для UI/API:

* endpoint `GET /indicators` возвращает “merged view”: hard defs + defaults.

---

## Ключевые решения

### 1) Hard defs в коде, UI defaults в YAML

* Hard defs неизменяемы по смыслу: `IndicatorDef / ParamDef / AxisDef / GridSpec` — это контракт.
* YAML — только параметры отображения/поиска (range/explicit), строго в рамках hard bounds.

### 2) Fail-fast валидация YAML на старте приложения

При старте API приложение обязано:

* загрузить registry (code defs),
* загрузить YAML defaults,
* валидировать соответствие,
* упасть с понятной ошибкой при нарушении правил.

### 3) Сохранённые комбинации — только explicit (без range)

Range/explicit в YAML нужны для grid-search и UI.
Сохранение “лучшего варианта” всегда делается в **explicit виде**:

* конкретные значения параметров,
* выбранные input series (если параметризуются).

### 4) Детерминированный `variant_key` v1 привязан к instrument + timeframe

Любая сохранённая конфигурация должна иметь стабильный ключ:

* одинаковый конфиг → одинаковый ключ,
* ключ зависит от `instrument_id` и `timeframe`,
* ключ зависит от набора индикаторов и их explicit параметров/inputs.

---

## Domain Definitions (code)

### Paths

`src/trading/contexts/indicators/domain/definitions/`

* `ma.py`
* `trend.py`
* `volatility.py`
* `momentum.py`
* `volume.py`
* `structure.py`

Каждый файл экспортирует **набор `IndicatorDef`** (или функцию `all_defs() -> Sequence[IndicatorDef]`), который затем объединяется в полный registry.

### Требования к каждому `IndicatorDef`

* `indicator_id` уникален и стабилен (используется в YAML, API, сохранениях);
* `inputs` перечислены явно (например `source: close|open|hlc3` если индикатор параметризуется источником);
* все параметры описаны через `ParamDef`:

  * hard bounds (`min/max`) и `step` (или enum values),
  * `ParamKind` (int/float/enum),
  * инварианты и человекочитаемые описания для UI (optional fields допустимы, но должны быть стабильны).

---

## YAML Defaults (configs/*/indicators.yaml)

### Paths

* `configs/dev/indicators.yaml`
* `configs/prod/indicators.yaml`
* `configs/test/indicators.yaml`

### Назначение YAML

YAML задаёт UI-friendly defaults для построения grid-search:

* `range` (start/stop_incl/step)
* `explicit` (values)

YAML **не задаёт hard bounds** и **не может их расширять**.

### Предлагаемый формат YAML v1

```yaml
schema_version: 1

defaults:
  ma.sma:
    inputs:
      source:
        mode: explicit
        values: ["close", "hlc3"]
    params:
      window:
        mode: range
        start: 5
        stop_incl: 200
        step: 1

  momentum.rsi:
    inputs:
      source:
        mode: explicit
        values: ["close"]
    params:
      window:
        mode: range
        start: 7
        stop_incl: 50
        step: 1
```

### YAML → Domain mapping

* `mode: explicit` → `ExplicitValuesSpec`
* `mode: range` → `RangeValuesSpec`
* Каждое поле в YAML должно соответствовать либо:

  * `ParamDef` (для `params`),
  * `InputSeries`/input axis (для `inputs`), если этот input параметризуем.

---

## Валидатор YAML (fail-fast rules)

### 1) Unknown indicator_id запрещён

Если в YAML встречается `defaults.<indicator_id>`, которого нет в code defs → ошибка.

### 2) Unknown param/input запрещён

Если YAML задаёт параметр/инпут, которого нет в `IndicatorDef` → ошибка.

### 3) Default spec должен быть внутри hard bounds

Для чисел:

* `start >= hard_min`
* `stop_incl <= hard_max`
* для `explicit`: каждое `value` в `[hard_min, hard_max]`

Для enum:

* values ⊆ hard allowed values

### 4) Step согласован с hard step

* Если hard step задан:

  * `yaml.step` должен быть кратен hard step (для int обычно `1`),
  * значения должны попадать на сетку шага.
* Для float — допускаем `eps` при проверке кратности/сетки.

### 5) Range должен быть валиден

* `start <= stop_incl`
* `step > 0`
* range должен генерировать хотя бы одно значение

### 6) Ограничение комбинаторики (guard)

EPIC-02 фиксирует **guard по комбинациям** для UI/search:

* базовый лимит Stage A combos (MA-only) фиксируем: **600_000**
* валидатор/конфиг-оценка должны уметь посчитать ожидаемое число комбинаций и fail-fast при превышении (для соответствующих индикаторов/осей).

*(Примечание: точный расчёт зависит от движка/оси, но правило “есть расчёт → есть guard → есть ошибка” фиксируется здесь.)*

---

## API: GET /indicators

### Purpose

UI получает “полный registry”:

* hard defs (bounds, step, типы, inputs),
* UI defaults из YAML (range/explicit specs).

### Contract (response shape, v1)

Ответ возвращает список индикаторов:

* `indicator_id`
* `group` (ma/trend/volatility/…)
* `inputs`:

  * hard allowed values
  * defaults spec (explicit/range)
* `params`:

  * hard bounds + kind + step/enum-values
  * defaults spec (explicit/range)
* `output_spec` / `layout` (если важно для UI)

### Paths / wiring

* `apps/api/routes/indicators.py` (новый роут)
* `apps/api/wiring/modules/indicators.py` (новый wiring module)
* YAML берём по окружению (аналогично `market_data.yaml`):

  * путь конфигов фиксирован: `configs/{env}/indicators.yaml`

---

## Application layer: Registry порт и реализация

### Port

Используем/расширяем существующий port `IndicatorRegistry` в:
`src/trading/contexts/indicators/application/ports/registry/indicator_registry.py`

Минимально нужно уметь:

* `list_all()` → merged view (hard defs + defaults)
* `get(indicator_id)` → merged view для одного индикатора (optional)

### Adapter (implementation)

Реализация будет в adapters (outbound/config или inbound/api — по текущим соглашениям проекта), но принцип такой:

* загружает code defs (`domain/definitions/*`)
* загружает YAML defaults (по env)
* валидирует (fail-fast)
* собирает merged DTO для API

---

## Совместимость: сохранение комбинаций и variant_key v1

### Что сохраняем

Сохраняем **explicit конфигурацию** индикаторов для конкретного инструмента и таймфрейма:

* `instrument_id` (shared_kernel `InstrumentId`)
* `timeframe` (shared_kernel `Timeframe`)
* `indicators[]`:

  * `indicator_id`
  * `inputs` (explicit)
  * `params` (explicit)

Range/explicit из YAML сюда не попадает — только конкретные значения.

### VariantKey v1 (детерминированный)

`variant_key = sha256(canonical_json(payload_v1))`

Где `payload_v1` — JSON со стабильным порядком ключей и стабильной сортировкой:

```json
{
  "schema_version": 1,
  "instrument_id": "<InstrumentId.as_str()>",
  "timeframe": "<Timeframe.as_str()>",
  "indicators": [
    {
      "id": "ma.sma",
      "inputs": [["source", "close"]],
      "params": [["window", 36]]
    }
  ]
}
```

Правила канонизации:

* `indicators` сортируются по `id` (строго);
* внутри `inputs` и `params` пары сортируются по ключу;
* числа сериализуются без потерь смысла (int как int, float как float; для float допускаем фиксированный формат, но главное — детерминизм).

### Валидация сохранённого конфига

При загрузке/использовании:

* `indicator_id` должен существовать в registry,
* значения inputs/params должны быть валидны по hard defs.

*(Persistence профиля/бандлов — следующий EPIC, но этот контракт уже обязателен для совместимости.)*

---

## DoD (EPIC-02)

1. В репозитории есть code defs:

* `src/trading/contexts/indicators/domain/definitions/*.py` по группам.

2. В репозитории есть YAML по окружениям:

* `configs/dev/indicators.yaml`
* `configs/prod/indicators.yaml`
* `configs/test/indicators.yaml`

3. На старте API происходит fail-fast валидация YAML:

* ошибки человекочитаемые,
* включают путь до поля YAML и причину (unknown id/param, out of bounds, step mismatch, guard exceeded).

4. Реализован `GET /indicators`:

* отдаёт полный registry (hard bounds + UI defaults),
* структура ответа стабильна и пригодна для UI.

5. Документ закрепляет `variant_key v1` с привязкой к `instrument_id + timeframe`.

---

Если ок — следующий шаг: я на базе этого документа сделаю машиночитаемый prompt для агента на реализацию EPIC-02 (как делали для EPIC-01), но ты просил пока без промта — так что жду “утверждаю документ / правки”.
