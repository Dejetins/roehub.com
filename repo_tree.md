# Карта репозитория Roehub

Старое статическое дерево заменено поддерживаемой картой проекта:

- текст и встроенная визуальная схема: [`docs/architecture/project-map/PROJECT_MAP.md`](docs/architecture/project-map/PROJECT_MAP.md);
- машиночитаемая карта для агентов: [`docs/architecture/project-map/project-map.json`](docs/architecture/project-map/project-map.json);
- правила навигации агентов и субагентов: [`docs/architecture/project-map/AGENT_GUIDE.md`](docs/architecture/project-map/AGENT_GUIDE.md).

Актуализация выполняется командой `python -m tools.docs.generate_project_map` и workflow `.github/workflows/update-project-map.yml` при каждом push.
