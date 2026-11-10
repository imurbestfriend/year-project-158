

# SAE decomposition

## Готовые SAE

Для open-source моделей можно посмотреть [тут](https://decoderesearch.github.io/SAELens/latest/pretrained_saes/).
Примеры работы с ними из библиотеки sae_lens есть в EDA.ipynb

## Хуки

Слой трансформера $i$ обычно содержит attention, потом MLP

Соответственно, в основном SAE разделяются на три типа, в зависимости от хука, который собирает активации:
- `hook_resid_pre`: смотрит на residual stream перед слоем $i$
- `hook_resid_mid`: смотрит на residual stream после прибавления результата attention слоя $i$, до MLP
- `hook_resid_post`: смотрит на residual stream после слоя $i$

## Интерпретация

Предположим, у нас уже есть пред-обученный SAE.
Собрав активации и применив SAE encode к ним, мы получаем разреженный вектор. Как интерпретировать активировавшиеся компоненты этого вектора?

| № | Подход | Идея | Статьи на arXiv | Реализации и примеры | Ограничение |
|---|---|---|---|---|---|
| 1 | **Примеры активации** | Собрать контексты срабатываний одной координаты SAE, подсветить токены и найти общий паттерн | [Cunningham et al., 2023 — Sparse Autoencoders Find Highly Interpretable Features in Language Models](https://arxiv.org/abs/2309.08600); [Gao et al., 2024 — Scaling and Evaluating Sparse Autoencoders](https://arxiv.org/abs/2406.04093) | [SAEDashboard](https://github.com/jbloomAus/SAEDashboard): визуализации примеров и статистик, совместимые с SAELens; [Neuronpedia](https://www.neuronpedia.org/): опубликованные признаки | Сильные срабатывания могут давать неполную картину; нужны умеренные активации и контрпримеры |
| 2 | **Автоматическая интерпретация — AutoInterp** | LLM формулирует объяснение по примерам; затем проверяется, помогает ли оно предсказывать срабатывания на новых текстах | [Paulo et al., 2024 — Automatically Interpreting Millions of Features in Large Language Models](https://arxiv.org/abs/2410.13928) | [Delphi](https://github.com/EleutherAI/delphi): сбор активаций, генерация и оценка объяснений; примеры запуска в README | Результат зависит от примеров и способа оценки; убедительное объяснение может плохо обобщаться |
| 3 | **Поиск заданного концепта: контрастные группы и sparse probing** | Сравнить примеры со свойством и без него; найти один или несколько SAE-признаков, предсказывающих свойство | [Karvonen et al., 2025 — SAEBench](https://arxiv.org/abs/2503.09532), раздел о sparse probing | [SAEBench: sparse_probing](https://github.com/adamkarvonen/SAEBench/tree/main/sae_bench/evals/sparse_probing): классификаторы на небольшом числе признаков, внутри scikit-learn | Предсказуемость свойства ещё не доказывает его использование моделью для генерации |
| 4 | **Анализ весов и проекция в словарь — logit lens** | Спроецировать decoder-направление через выходную матрицу модели и изучить токены с большими положительными и отрицательными значениями | Общая идея и ограничения: [Belrose et al., 2023 — Eliciting Latent Predictions from Transformers with the Tuned Lens](https://arxiv.org/abs/2303.08112); более широкий анализ SAE по весам: [Liu & Deng, 2026 — What Does a Sparse Autoencoder Feature Do?](https://arxiv.org/abs/2601.22447) | [SAELens: logits_lens_with_features.ipynb](https://github.com/decoderesearch/SAELens/blob/main/tutorials/logits_lens_with_features.ipynb): анализ SAE-направлений через выходную матрицу | Прямая проекция не учитывает все последующие вычисления; это подсказка, а не измеренный эффект интервенции |
| 5 | **Каузальная проверка: абляция, усиление, подмена** | Изменить активацию признака и измерить изменения предсказаний или ответов | [Cunningham et al., 2023](https://arxiv.org/abs/2309.08600); [Arad et al., 2025 — SAEs Are Good for Steering — If You Select the Right Features](https://arxiv.org/abs/2505.20063) | [SAELens: using_an_sae_as_a_steering_vector.ipynb](https://github.com/decoderesearch/SAELens/blob/main/tutorials/using_an_sae_as_a_steering_vector.ipynb): учебный пример steering | Смысл входных срабатываний может расходиться с эффектом на выходе; результат зависит от силы и места вмешательства |

## Интервенция

Предположим, у нас уже есть SAE и мы выбрали интерпретируемые признаки. Как через них повлиять на генерацию без изменения весов модели?

Пусть $x$ — активация модели в хуке SAE, $z_i$ — активация признака, $d_i$ — его decoder-направление. Для линейного декодера изменение признака с $z_i$ на $z_i'$ можно внести как $x' = x + (z_i' - z_i)d_i$. Так мы сохраняем исходную ошибку реконструкции SAE, а не заменяем всю активацию её приближением.

| № | Подход | Идея | Статьи и разборы | Реализации и примеры | Ограничение |
|---|---|---|---|---|---|
| 1 | **Добавление направления — additive steering** | Прибавлять $\alpha d_i$ к активации модели: положительный коэффициент добавляет направление, отрицательный — вычитает | [Arad et al., 2025 — SAEs Are Good for Steering — If You Select the Right Features](https://arxiv.org/abs/2505.20063) | [SAELens: using_an_sae_as_a_steering_vector.ipynb](https://github.com/decoderesearch/SAELens/blob/main/tutorials/using_an_sae_as_a_steering_vector.ipynb): добавление decoder-направления через хук | Одинаковая добавка при разной исходной активации; вычитание направления не обязательно вызывает противоположное поведение |
| 2 | **Масштабирование и абляция** | Задать $z_i' = \beta z_i$: усилить при $\beta > 1$, ослабить при $0 < \beta < 1$, обнулить при $\beta = 0$ | [Cunningham et al., 2023](https://arxiv.org/abs/2309.08600): абляции SAE-признаков | Хук с encode и добавкой $(\beta - 1)z_i d_i$; обнуление проверяет, нужен ли активный признак для ответа | Не включает признак, если он не сработал; свойство может поддерживаться несколькими признаками |
| 3 | **Фиксация активации — clamping** | Задать $z_i' = c$ независимо от исходного значения; $c = 0$ также даёт абляцию | [Templeton et al., 2024 — Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html): управление генерацией через фиксацию признаков | Хук с encode и добавкой $(c - z_i)d_i$; пример — усиление признака Golden Gate Bridge в Claude | Постоянное значение может быть неестественным для контекста; слишком сильное вмешательство ухудшает связность и релевантность |

Можно менять несколько признаков сразу, суммируя добавки. Отдельно выбираем **где и когда вмешиваться**: слой и хук SAE, позиции токенов, обработка промпта и/или генерация новых токенов. Интерпретация срабатываний ещё не гарантирует нужного эффекта на выходе: сравниваем с генерацией без вмешательства, подбираем силу и проверяем целевое свойство вместе с качеством ответа.
