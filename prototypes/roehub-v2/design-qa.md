# Проверка соответствия дизайну — Roehub v2

## Артефакты сравнения

- пользовательский снимок с отмеченным дублированием навигации: `/Users/daniildegtyarev/Projects/roehub.com/prototypes/roehub-v2/evidence/strategies-user-reference.png`;
- нормализованный Strategies до исправления при `1189x611`: `/Users/daniildegtyarev/Projects/roehub.com/prototypes/roehub-v2/evidence/strategies-before-single-nav-1189x611.png`;
- итоговый Strategies при `1189x611`: `/Users/daniildegtyarev/Projects/roehub.com/prototypes/roehub-v2/evidence/strategies-after-single-nav-1189x611.png`;
- полноэкранное парное сравнение навигации и сетки: `/Users/daniildegtyarev/Projects/roehub.com/prototypes/roehub-v2/evidence/strategies-single-nav-comparison-1189x611.jpg`;
- фокусное сравнение левой навигации: `/Users/daniildegtyarev/Projects/roehub.com/prototypes/roehub-v2/evidence/navigation-single-nav-focus-comparison-1189x611.jpg`;
- итоговый Strategies при `1440x1024`: `/Users/daniildegtyarev/Projects/roehub.com/prototypes/roehub-v2/evidence/strategies-single-nav-desktop-1440x1024.jpg`;
- итоговый мобильный Strategies при `390x844`: `/Users/daniildegtyarev/Projects/roehub.com/prototypes/roehub-v2/evidence/strategies-single-nav-mobile-390x844.png`;
- исходный Overview до доработки пространства: `/Users/daniildegtyarev/Projects/roehub.com/prototypes/roehub-v2/evidence/overview-before-spacing.jpg`;
- обновлённый Overview: `/Users/daniildegtyarev/Projects/roehub.com/prototypes/roehub-v2/evidence/overview-desktop-spacing-v2.jpg`;
- парное сравнение Overview до и после: `/Users/daniildegtyarev/Projects/roehub.com/prototypes/roehub-v2/evidence/overview-spacing-comparison.jpg`;
- парное сравнение Settings до и после: `/Users/daniildegtyarev/Projects/roehub.com/prototypes/roehub-v2/evidence/settings-spacing-comparison.jpg`;
- парное сравнение Models до и после: `/Users/daniildegtyarev/Projects/roehub.com/prototypes/roehub-v2/evidence/models-spacing-comparison.jpg`;
- обновлённые мобильные экраны: `evidence/overview-mobile-spacing-v2.jpg`, `evidence/strategies-mobile-spacing-v2.jpg`, `evidence/models-mobile-spacing-v2.jpg`, `evidence/settings-mobile-spacing-v2.jpg`;
- визуальный эталон: `/Users/daniildegtyarev/Projects/roehub.com/prototypes/roehub-v2/evidence/visual-target-1440x1024.jpg`;
- первая браузерная реализация: `/Users/daniildegtyarev/Projects/roehub.com/prototypes/roehub-v2/evidence/overview-desktop-pass1.jpg`;
- промежуточная реализация: `/Users/daniildegtyarev/Projects/roehub.com/prototypes/roehub-v2/evidence/overview-desktop-pass2.jpg`;
- итоговая браузерная реализация: `/Users/daniildegtyarev/Projects/roehub.com/prototypes/roehub-v2/evidence/overview-desktop-final.jpg`;
- фокусный эталон центральной рабочей области: `/Users/daniildegtyarev/Projects/roehub.com/prototypes/roehub-v2/evidence/reference-focus.jpg`;
- фокусная итоговая реализация: `/Users/daniildegtyarev/Projects/roehub.com/prototypes/roehub-v2/evidence/implementation-focus.jpg`;
- итоговый планшет: `/Users/daniildegtyarev/Projects/roehub.com/prototypes/roehub-v2/evidence/overview-tablet.jpg`;
- итоговый мобильный Overview: `/Users/daniildegtyarev/Projects/roehub.com/prototypes/roehub-v2/evidence/overview-mobile.jpg`;
- мобильный Backtests: `/Users/daniildegtyarev/Projects/roehub.com/prototypes/roehub-v2/evidence/backtests-mobile.jpg`;
- светлая тема: `/Users/daniildegtyarev/Projects/roehub.com/prototypes/roehub-v2/evidence/settings-paper-mobile.jpg`;
- мобильный Login: `/Users/daniildegtyarev/Projects/roehub.com/prototypes/roehub-v2/evidence/login-mobile.jpg`;
- карта программы после исправления deep-link tab: `/Users/daniildegtyarev/Projects/roehub.com/prototypes/roehub-v2/evidence/program-map-mobile-final.jpg`.

Основное исходное сравнение выполнено при одинаковом viewport `1440x1024`, состоянии `Overview`, теме `graphite`, выбранной стратегии `RL Alpha v14`, открытом inspector и раскрытом activity drawer. Текущая итерация навигации и Strategies дополнительно сравнена при точном пользовательском viewport `1189x611`, теме `graphite` и раскрытом activity drawer.

## Проверка обновлённого пространственного ритма

Исходной визуальной правдой для этой итерации служат браузерные снимки до изменения: `overview-before-spacing.jpg`, `settings-before-spacing.jpg` и `models-before-spacing.jpg`. Реализация сравнивалась с ними в объединённых изображениях, где исходная версия находится слева, а обновлённая — справа.

Полноэкранное сравнение `overview-spacing-comparison.jpg` подтверждает намеренные изменения без смены визуального характера: рабочая область стала шире, внешние поля страницы увеличены, а расстояние между KPI, графиком и модулями приведено к единому ритму `12px`.

Фокусные сравнения `settings-spacing-comparison.jpg` и `models-spacing-comparison.jpg` показывают исправление двух наиболее заметных пустых зон:

- `Workspace defaults` теперь занимает всю строку и размещает три поля в равных колонках;
- `Promotion contract` в Models занимает всю нижнюю строку, поэтому служебные проверки распределяются горизонтально, а большая пустая область в левом модуле существенно сокращена;
- пятая карточка Strategies растягивается на оставшиеся две колонки последнего ряда вместо пустой ячейки;
- Connections получает более свободные строки провайдеров и увеличенные внутренние поля без изменения информационной архитектуры.

Дополнительные фокусные фрагменты не потребовались: типографика, границы, состояния, значения и подписи читаются в исходном масштабе объединённых сравнений.

## Полноэкранное сравнение

Эталон и итоговая реализация были открыты одновременно в одном сравнении при исходном разрешении. Итог сохраняет ключевую композицию:

- единую основную навигацию слева без дублирующего global rail;
- top toolbar и document tabs;
- основную рабочую плоскость с KPI, chart и четырьмя data modules;
- right inspector выбранного объекта;
- activity drawer и status bar;
- спокойную графитовую палитру без orange terminal framing.

## Фокусное сравнение

Центральные области обоих изображений дополнительно сравнены в crop `1100x650`. Текст таблиц, ритм строк, состояние badges, chart legend, inspector sections, borders и surface hierarchy читаются в исходном масштабе. Отдельные дополнительные crops не потребовались: все пять обязательных fidelity surfaces различимы в полном и фокусном сравнении.

## История исправлений

### Итерация 1 — результат заблокирован

- [P1] Пропорции оболочки и рабочей области не соответствовали эталону.
  - Доказательство: в `evidence/overview-desktop-pass1.jpg` context navigation была шире, top/document bars выше, а chart и таблицы занимали больше вертикального пространства, чем в эталоне.
  - Исправление: rail изменен до `46px`, context navigation до `174px`, inspector до `264px`; top bar и document tabs уменьшены; chart и table density откалиброваны.
- [P1] Сравнивались разные состояния программы.
  - Доказательство: в первой реализации inspector показывал общий workspace, activity drawer был свернут; эталон показывал выбранную `RL Alpha v14` и раскрытую активность.
  - Исправление: стартовый state синхронизирован с эталоном, inspector дополнен performance, readiness и links.

Постпроверка: `evidence/overview-desktop-pass2.jpg`.

### Итерация 2 — результат заблокирован

- [P2] При прямом mobile deep link активный document tab мог отсутствовать или оставаться за видимой областью.
  - Доказательство: `evidence/program-map-before-tab-fix.jpg` не показывал активную вкладку Program map.
  - Исправление: initial tab set теперь строится из текущего route, а навигация перемещает активную вкладку первой.
  - Постпроверка: `evidence/program-map-mobile-final.jpg`, активная вкладка — `Program map`.
- [P2] Переход из notification center оставлял overlay открытым.
  - Исправление: выбор уведомления теперь выполняет navigation и закрывает notification center одним действием.
- [P2] Inspector мог перекрывать мобильную поверхность после смены breakpoint.
  - Исправление: при ширине `≤820px` inspector закрывается; мобильная итоговая проверка подтверждает `display: none` и отсутствие horizontal overflow.

Постпроверка: desktop `evidence/overview-desktop-final.jpg`, tablet `evidence/overview-tablet.jpg`, mobile `evidence/overview-mobile.jpg`.

### Итерация 3 — результат заблокирован до исправлений

- [P2] Межмодульный ритм оставался слишком плотным, а часть сеток оставляла визуально случайные пустые ячейки.
  - Доказательство: `evidence/overview-before-spacing.jpg`, `evidence/settings-before-spacing.jpg`, `evidence/models-before-spacing.jpg`.
  - Исправление: введены общие токены `--workspace-gap`, `--page-gutter-x`, `--page-gutter-y` и `--module-padding`; рабочие промежутки увеличены с `8px` до `12px`; оболочка откалибрована до `164px` для контекстной навигации и `236px` для инспектора.
- [P2] Settings и Strategies оставляли незаполненную ячейку в последнем ряду, а Models концентрировал слишком много пустого пространства внутри одного блока.
  - Исправление: `Workspace defaults` и `Promotion contract` растянуты на всю строку; последняя карточка Strategies занимает остаток трёхколоночной сетки.
- [P2] Чистая браузерная проверка выявила вложенную кнопку в карточке стратегии и повторяющийся ключ `BTCUSDT` в таблице позиций.
  - Исправление: карточка разделена на семантический контейнер и отдельную основную кнопку; `Table` поддерживает функцию вычисления ключа, а позиции используют составной ключ.

Постпроверка: `evidence/overview-desktop-spacing-v2.jpg`, `evidence/overview-mobile-spacing-v2.jpg`, `evidence/settings-spacing-comparison.jpg`, `evidence/models-spacing-comparison.jpg`. Ошибки и предупреждения консоли в чистом повторном проходе отсутствуют.

### Итерация 4 — результат заблокирован до исправлений

- [P1] Две левые панели одновременно представляли основную навигацию и дублировали одни и те же разделы.
  - Доказательство: пользовательский маркер в `evidence/strategies-user-reference.png` и нормализованный снимок `evidence/strategies-before-single-nav-1189x611.png` показывают global rail шириной `46px` рядом с context navigation шириной `54px`.
  - Исправление: global rail удалён; context navigation стала единственным семантическим `nav` и получила бренд, разделы, командный поиск, уведомления, тему и пользовательское меню. По умолчанию панель занимает `64px` и раскрывается по запросу.
- [P2] Strategies использовала жёсткий трёхколоночный ритм, из-за чего при промежуточной ширине карточки оставались тесными, а изменение размеров происходило скачком.
  - Исправление: страница получила контейнерную адаптацию, текучую `flex`-сетку, масштабируемые внутренние поля и устойчивый переход `3 → 2 → 1`. При `1440px` исключён вариант из четырёх карточек и одной чрезмерно растянутой карточки.
- [P2] Свёрнутые инструменты единственной панели полагались только на иконки.
  - Исправление: всем разделам, инструментам и пользовательскому меню заданы явные доступные имена; отдельная клавиатурная проверка показала outline `2px` с отступом `2px`.

Постпроверка: полноэкранное сравнение `evidence/strategies-single-nav-comparison-1189x611.jpg`, фокусное сравнение `evidence/navigation-single-nav-focus-comparison-1189x611.jpg`, desktop `evidence/strategies-single-nav-desktop-1440x1024.jpg` и mobile `evidence/strategies-single-nav-mobile-390x844.png`. При `1440x1024` и `1189x611` карточки образуют ряды `3 + 2`, при `1024x768` и `820x800` — `2 + 2 + 1`, при `390x844` — одну колонку. Глобального горизонтального переполнения нет.

## Обязательные поверхности соответствия

### Шрифты и типографика

- системный sans-serif последовательно используется для интерфейса и данных;
- hierarchy соответствует эталону: page title, module title, metric, table heading, metadata;
- мелкий текст не перекрывается, длинные document tabs ограничены и прокручиваются;
- mono-глифы используются только в `kbd`, а не как общий terminal motif.

### Отступы и ритм компоновки

- shell использует одну левую панель и оставляет больше ширины центральной рабочей области;
- modules используют единый `5px` radius, тонкие borders, межблочный ритм `12px` и увеличенные внутренние поля;
- desktop, tablet и mobile не имеют глобального horizontal overflow;
- activity drawer не закрывает persistent navigation или status bar.

### Цвета и tokens

- `graphite` соответствует спокойному темному baseline;
- positive, negative, warning и info разведены семантически;
- проверена яркая тема `paper`; дополнительно реализованы `abyss`, `slate`, `frost`, `sand`;
- для всех шести тем автоматизированные пары `text`, `text-soft`, `text-muted`, `text-faint` и `accent-strong` относительно основных surfaces дают не менее `4.74:1` в проверенной матрице;
- orange не используется как универсальная рамка или декоративное свечение.

### Качество изображений и assets

- растровые изображения и декоративная imagery в целевой рабочей области отсутствуют;
- charts отрисованы библиотекой Recharts, icons — единым набором Phosphor;
- inline SVG, handcrafted SVG, emoji, CSS illustrations и placeholder imagery не используются;
- mark в shell является библиотечным интерфейсным icon, а не заявкой на production-логотип; официальный Roehub brand asset в исходной системе не найден.

### Текст и содержание

- copy описывает самостоятельную программу, а не процесс создания прототипа;
- все основные сущности и действия имеют реалистичные trading/research labels;
- auth copy явно объясняет границу Keycloak и не обещает локальное хранение пароля.

### Доступность и состояния

- единственная основная навигация объявлена семантически, а её свёрнутые controls имеют доступные имена, focus-visible outline и reduced-motion режим;
- charts имеют текстовое описание и table fallback;
- реализованы loading, empty, error/retry, stale, restricted, disabled и success;
- мобильная навигация имеет отдельную нативную модель с нижней панелью и sheet.

## Открытые вопросы

- Перед production-переносом нужен утвержденный vector brand mark; текущий Phosphor icon является нейтральным интерфейсным обозначением.
- Production typography следует повторно проверить с фактическим набором шрифтов текущего gateway и локалями `en`/`ru`.

## Остаточная полировка P3

- При production-переносе можно добавить более детальную внутридневную вариативность portfolio series и утвержденный brand mark без изменения layout contract.

## Итог

Actionable `P0`, `P1` и `P2` различий не осталось. Браузерные доказательства, объединённые сравнения до и после, взаимодействия, проверка консоли и адаптивные состояния подтверждены.

final result: passed
